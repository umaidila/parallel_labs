#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>


using namespace std;


void multiply_scalar(double* A, size_t colsA, size_t rowsA,
    const double* B, size_t colsB, size_t rowsB,
    const double* C, size_t colsC, size_t rowsC)
{
    assert(colsB == rowsC && colsA == colsC && rowsA == rowsB);

    for (size_t col = 0; col < colsA; ++col)
    {
        for (size_t row = 0; row < rowsA; ++row)
        {
            A[col * rowsA + row] = 0;
            for (size_t k = 0; k < colsB; ++k)
            {
                A[col * rowsA + row] += B[k * rowsB + row] * C[col * rowsC + k];
            }
        }
    }
}


void multiply_avx(double* A, size_t colsA, size_t rowsA,
    const double* B, size_t colsB, size_t rowsB,
    const double* C, size_t colsC, size_t rowsC)
{
    assert(colsB == rowsC && colsA == colsC && rowsA == rowsB);

    for (size_t rowBlock = 0; rowBlock < rowsB / 4; ++rowBlock)
    {
        for (size_t col = 0; col < colsC; ++col)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rowsC; ++k)
            {
                __m256d bVec = _mm256_loadu_pd(B + k * rowsB + rowBlock * 4);
                __m256d cVal = _mm256_set1_pd(C[col * rowsC + k]);
                sum = _mm256_fmadd_pd(bVec, cVal, sum);
            }
            _mm256_storeu_pd(A + col * rowsA + rowBlock * 4, sum);
        }
    }
}


void randomize_matrix(double* matrix, std::size_t order)
{
    std::uniform_real_distribution<double> distribution(0.0, 100000.0);
    std::default_random_engine generator;
    for (std::size_t i = 0; i < order * order; ++i)
    {
        matrix[i] = distribution(generator);
    }
}

void display_matrix(const double* matrix, std::size_t cols, std::size_t rows)
{
    for (std::size_t row = 0; row < rows; ++row)
    {
        std::cout << "[" << matrix[row * cols];
        for (std::size_t col = 1; col < cols; ++col)
        {
            std::cout << ", " << matrix[row * cols + col];
        }
        std::cout << "]\n";
    }
}

int main()
{
    const std::size_t matrixOrder = 16 * 4 * 9;
    const std::size_t experimentCount = 10;

    std::ofstream output("output3.csv");
    if (!output.is_open())
    {
        std::cerr << "Error: Unable to open output file!\n";
        return 1;
    }

    vector<double> A(matrixOrder * matrixOrder),
        C(matrixOrder * matrixOrder),
        D(matrixOrder * matrixOrder),
        E(matrixOrder * matrixOrder);
    vector<double> B(matrixOrder * matrixOrder, 1.0);
    

    randomize_matrix(A.data(), matrixOrder);

    multiply_scalar(C.data(), matrixOrder, matrixOrder,
        A.data(), matrixOrder, matrixOrder,
        B.data(), matrixOrder, matrixOrder);

    multiply_avx(D.data(), matrixOrder, matrixOrder,
        A.data(), matrixOrder, matrixOrder,
        B.data(), matrixOrder, matrixOrder);

    
    if (memcmp(C.data(), D.data(), matrixOrder * matrixOrder * sizeof(double)) != 0 ||
        memcmp(C.data(), E.data(), matrixOrder * matrixOrder * sizeof(double)) != 0)
    {
        std::cerr << "correct test failed\n";
        output.close();
        return 1;
    }
    std::cout << "correct test passed\n";

   
    output << "T,Duration,Speedup\n";

    double scalarTime = 0;
    for (std::size_t i = 0; i < experimentCount; ++i)
    {
        randomize_matrix(A.data(), matrixOrder);
        auto start = std::chrono::steady_clock::now();
        multiply_scalar(C.data(), matrixOrder, matrixOrder,
            A.data(), matrixOrder, matrixOrder,
            B.data(), matrixOrder, matrixOrder);
        auto end = std::chrono::steady_clock::now();
        scalarTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    scalarTime /= experimentCount;
    std::cout << "Scalar multiplication: " << scalarTime << " ms, Speedup = 1.0\n";
    output << "Scalar," << scalarTime << ",1.0\n";

    double avx2Time = 0;
    for (std::size_t i = 0; i < experimentCount; ++i)
    {
        randomize_matrix(A.data(), matrixOrder);
        auto start = std::chrono::steady_clock::now();
        multiply_avx(D.data(), matrixOrder, matrixOrder,
            A.data(), matrixOrder, matrixOrder,
            B.data(), matrixOrder, matrixOrder);
        auto end = std::chrono::steady_clock::now();
        avx2Time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    avx2Time /= experimentCount;
    std::cout << "AVX2 multiplication: " << avx2Time << " ms, Speedup = " << scalarTime / avx2Time << "\n";
    output << "AVX2," << avx2Time << "," << scalarTime / avx2Time << "\n";

    output.close();
    return 0;
}
