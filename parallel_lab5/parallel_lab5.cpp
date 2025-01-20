#include <barrier>
#include <bit>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

static unsigned nibble[16] = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 };

unsigned flip_b(unsigned byte) {
    return (nibble[byte & 15] << 4) | nibble[(byte >> 4) & 15];
}

unsigned flip_s(unsigned v) {
    return flip_b(v & 0xFF) << 8 | flip_b(v >> 8);
}

unsigned flip_i(unsigned v) {
    return flip_s(v & 0xFFFF) << 16 | flip_s(v >> 16);
}

unsigned long long flip_ll(unsigned long long v) {
    return (unsigned long long)flip_i(v & 0xFFFFFFFF) << 32 | flip_i(v >> 32);
}

void bit_shuffle(const std::complex<double>* inp, std::complex<double>* out, std::size_t n) {
    std::size_t shift = std::countl_zero(n) + 1;
    for (std::size_t i = 0; i < n; i++) {
        out[flip_ll(i) >> shift] = inp[i];
    }
}

struct thread_range {
    std::size_t start;
    std::size_t end;
};

thread_range thread_task_range(std::size_t task_count, std::size_t thread_count, std::size_t thread_id) {
    auto extra = task_count % thread_count;
    auto base_size = task_count / thread_count;
    auto start = thread_id < extra ? (base_size + 1) * thread_id : base_size * thread_id + extra;
    auto end = start + (thread_id < extra ? base_size + 1 : base_size);
    return { start, end };
}

void fft_nonrec_multithreaded_core(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, int inverse, std::size_t thread_count) {
    bit_shuffle(inp, out, n);
    std::barrier<> sync_point(thread_count);

    auto worker = [&out, n, inverse, thread_count, &sync_point](std::size_t thread_id) {
        for (std::size_t group_length = 2; group_length <= n; group_length <<= 1) {
            auto [start, end] = thread_task_range(n / group_length, thread_count, thread_id);

            for (std::size_t group = start; group < end; group++) {
                for (std::size_t i = 0; i < group_length / 2; i++) {
                    auto w = std::polar(1.0, -2 * std::numbers::pi_v<double> *i * inverse / group_length);
                    auto r1 = out[group_length * group + i];
                    auto r2 = out[group_length * group + i + group_length / 2];
                    out[group_length * group + i] = r1 + w * r2;
                    out[group_length * group + i + group_length / 2] = r1 - w * r2;
                }
            }

            sync_point.arrive_and_wait();
        }
        };

    std::vector<std::thread> threads(thread_count - 1);
    for (std::size_t i = 1; i < thread_count; i++) {
        threads[i - 1] = std::thread(worker, i);
    }
    worker(0);

    for (auto& t : threads) {
        t.join();
    }
}

void fft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count) {
    fft_nonrec_multithreaded_core(inp, out, n, 1, thread_count);
}

void ifft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count) {
    fft_nonrec_multithreaded_core(inp, out, n, -1, thread_count);
    for (std::size_t i = 0; i < n; i++) {
        out[i] /= static_cast<std::complex<double>>(n);
    }
}

int main() {
    const std::size_t exp_count = 10;
    constexpr std::size_t n = 1llu << 20;
    std::vector<std::complex<double>> original(n), spectre(n), restored(n);

    auto randomize_vector = [](std::vector<std::complex<double>>& v) {
        std::uniform_real_distribution<double> unif(0.0, 100000.0);
        std::default_random_engine re;
        for (auto& elem : v) {
            elem = unif(re);
        }
        };

    auto approx_equal = [](const std::vector<std::complex<double>>& v1, const std::vector<std::complex<double>>& v2) {
        for (std::size_t i = 0; i < v1.size(); i++) {
            if (std::abs(v1[i] - v2[i]) > 0.0001) {
                return false;
            }
        }
        return true;
        };

    std::ofstream output("output.csv");
    if (!output.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    output << "T,Duration,Speedup\n";

    double base_time = 0; // Для времени выполнения с одним потоком

    for (std::size_t i = 1; i <= std::thread::hardware_concurrency(); i++) {
        double total_time = 0;
        for (std::size_t j = 0; j < exp_count; j++) {
            randomize_vector(original);
            auto start = std::chrono::steady_clock::now();
            fft_nonrec_multithreaded(original.data(), spectre.data(), n, i);
            auto end = std::chrono::steady_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }

        double average_time = total_time / exp_count;

        if (i == 1) {
            base_time = average_time; // Сохраняем время для одного потока
        }

        double speedup = base_time / average_time; // Расчёт ускорения
        std::cout << "FFT: Threads = " << i << ", Avg. Duration = " << average_time << " ms, Speedup = " << speedup << "\n";
        output << i << "," << average_time << "," << speedup << "\n";
    }


    output.close();
    return 0;
}
