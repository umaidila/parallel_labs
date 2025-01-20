#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <immintrin.h>
#include <vector>


#define COLUMNS 2048 * 2
#define ROWS 2048 * 2

void matrix_addition(double* result, const double* mat1, const double* mat2, size_t num_cols, size_t num_rows)
{
    for (size_t i = 0; i < num_cols * num_rows; ++i)
    {
        result[i] = mat1[i] + mat2[i];
    }
}

void matrix_addition_avx(double* result, const double* mat1, const double* mat2, size_t num_cols, size_t num_rows)
{
    for (size_t i = 0; i < num_cols * num_rows / 4; ++i)
    {
        __m256d vec1 = _mm256_loadu_pd(&mat1[i * 4]);
        __m256d vec2 = _mm256_loadu_pd(&mat2[i * 4]);
        __m256d sum = _mm256_add_pd(vec1, vec2);
        _mm256_storeu_pd(&result[i * 4], sum);
    }
}

void matrix_addition_avx512(double* result, const double* mat1, const double* mat2, size_t num_cols, size_t num_rows)
{
    for (size_t i = 0; i < num_cols * num_rows / 8; ++i)
    {
        __m512d vec1 = _mm512_loadu_pd(&mat1[i * 8]);
        __m512d vec2 = _mm512_loadu_pd(&mat2[i * 8]);
        __m512d sum = _mm512_add_pd(vec1, vec2);
        _mm512_storeu_pd(&result[i * 8], sum);
    }
}

int main()
{
    const std::size_t EPOCHS = 10;

    std::ofstream output_file("output.csv");
    if (!output_file.is_open())
    {
        std::cerr << "error opening output file\n";
        return 1;
    }

    std::vector<double> mat1(COLUMNS * ROWS, 1.0), mat2(COLUMNS * ROWS, -1.0), result(COLUMNS * ROWS, -0.1);

    auto display_matrix = [](const double* matrix, size_t num_cols, size_t num_rows)
        {
            for (size_t row = 0; row < num_rows; ++row)
            {
                std::cout << "[" << matrix[row * num_cols];
                for (size_t col = 1; col < num_cols; ++col)
                {
                    std::cout << ", " << matrix[row * num_cols + col];
                }
                std::cout << "]\n";
            }
            std::cout << "\n";
        };


    matrix_addition_avx512(result.data(), mat1.data(), mat2.data(), COLUMNS, ROWS);
    for (std::size_t i = 0; i < COLUMNS * ROWS; ++i)
    {
        if (result[i] != 0.0)
        {
            std::cerr << "correct test failed\n";
            return 1;
        }
    }
    std::cout << "correct test passed\n";

    output_file << "T,Duration,speedup\n";

    // Scalar 
    double scalar_avg_time = 0.0;
    for (std::size_t i = 0; i < EPOCHS; ++i)
    {
        auto start_time = std::chrono::steady_clock::now();
        matrix_addition(result.data(), mat1.data(), mat2.data(), COLUMNS, ROWS);
        auto end_time = std::chrono::steady_clock::now();
        scalar_avg_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
    scalar_avg_time /= EPOCHS;

    std::cout << "scalar addition: " << scalar_avg_time << " ms, speedup = 1.0\n";
    output_file << "scalar," << scalar_avg_time << ",1.0\n";

    // AVX 
    double avx_avg_time = 0.0;
    for (std::size_t i = 0; i < EPOCHS; ++i)
    {
        auto start_time = std::chrono::steady_clock::now();
        matrix_addition_avx(result.data(), mat1.data(), mat2.data(), COLUMNS, ROWS);
        auto end_time = std::chrono::steady_clock::now();
        avx_avg_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
    avx_avg_time /= EPOCHS;

    std::cout << "AVX addition: " << avx_avg_time << " ms, speedup = " << scalar_avg_time / avx_avg_time << "\n";
    output_file << "AVX," << avx_avg_time << "," << scalar_avg_time / avx_avg_time << "\n";

  

    // AVX512
    double avx512_avg_time = 0.0;
    for (std::size_t i = 0; i < EPOCHS; ++i)
    {
        auto start_time = std::chrono::steady_clock::now();
        matrix_addition_avx512(result.data(), mat1.data(), mat2.data(), COLUMNS, ROWS);
        auto end_time = std::chrono::steady_clock::now();
        avx512_avg_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
    avx512_avg_time /= EPOCHS;

    std::cout << "AVX512 addition: " << avx512_avg_time << " ms, speedup = " << scalar_avg_time / avx512_avg_time << "\n";
    output_file << "AVX512," << avx512_avg_time << "," << scalar_avg_time / avx512_avg_time << "\n";

  
    output_file.close();
    
    return 0;
}
