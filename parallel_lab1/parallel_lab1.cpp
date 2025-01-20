#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>

const double STEPS = 100000000;

double my_function(double x)
{
    return x * x;
}

double integral_serial(double start, double end)
{
    double total = 0;
    double step_size = (end - start) / STEPS;

    for (std::size_t i = 0; i < STEPS; ++i)
    {
        total += my_function(start + i * step_size);
    }

    return step_size * total;
}

double integral_parallel(double start, double end)
{
    double total = 0;
    double step_size = (end - start) / STEPS;

#pragma omp parallel
    {
        unsigned thread_id = omp_get_thread_num();
        unsigned thread_count = omp_get_num_threads();
        double local_res = 0;

        for (std::size_t i = thread_id; i < STEPS; i += thread_count)
        {
            local_res += my_function(start + i * step_size);
        }

#pragma omp critical
        {
            total += local_res;
        }
    }

    return step_size * total;
}

int main()
{
    std::ofstream file("output.csv");
    if (!file)
    {
        std::cerr << "Failed to open file!\n";
        return 1;
    }

    file << "T,Duration,Speedup\n";

    double start_time = omp_get_wtime();
    double serial_result = integral_serial(-1, 1);
    double serial_duration = omp_get_wtime() - start_time;

    std::cout << "Serial: Threads = 1, Result = " << serial_result
        << ", Time = " << serial_duration << "s, Speedup = 1.0\n";
    file << "1," << serial_duration << ",1.0\n";

    for (std::size_t threads = 2; threads <= std::thread::hardware_concurrency(); ++threads)
    {
        omp_set_num_threads(threads);
        start_time = omp_get_wtime();
        double parallel_result = integral_parallel(-1, 1);
        double parallel_duration = omp_get_wtime() - start_time;

        std::cout << "Parallel: Threads = " << threads << ", Result = " << parallel_result
            << ", Time = " << parallel_duration << "s, Speedup = " << serial_duration / parallel_duration << "\n";
        file << threads << "," << parallel_duration << "," << (serial_duration / parallel_duration) << "\n";
    }

    file.close();
    return 0;
}
