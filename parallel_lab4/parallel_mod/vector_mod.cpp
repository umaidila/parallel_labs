#include "vector_mod.h"
#include "mod_ops.h"
#include "num_threads.h"
#include <thread>
#include <vector>
#include <barrier>
#include <cstdint>


IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
    IntegerWord result = 1;
    while (power > 0) {
        if (power % 2 != 0) {
            result = mul_mod(result, base, mod);
        }
        power >>= 1;
        base = mul_mod(base, base, mod);
    }
    return result;
}


IntegerWord word_pow_mod(size_t power, IntegerWord mod) {
    return pow_mod((-mod) % mod, power, mod);
}


struct thread_range {
    std::size_t start;
    std::size_t end;
};


thread_range vector_thread_range(size_t total_elements, unsigned num_threads, unsigned thread_id) {
    auto extra = total_elements % num_threads;
    auto base_size = total_elements / num_threads;
    auto start = thread_id < extra ? (base_size + 1) * thread_id : base_size * thread_id + extra;
    auto end = start + (thread_id < extra ? base_size + 1 : base_size);
    return { start, end };
}


struct partial_result_t {
    alignas(std::hardware_destructive_interference_size) IntegerWord value;
};


IntegerWord vector_mod(const IntegerWord* V, std::size_t N, IntegerWord mod) {
    size_t num_threads = get_num_threads();
    std::vector<std::thread> threads(num_threads - 1);
    std::vector<partial_result_t> partial_results(num_threads);
    std::barrier<> barrier(num_threads);

    auto worker = [V, N, num_threads, mod, &partial_results, &barrier](unsigned thread_id) {
        auto [start, end] = vector_thread_range(N, num_threads, thread_id);

        IntegerWord sum = 0;

        for (std::size_t i = end; i > start;) {
            sum = add_mod(mul_mod(sum, -mod, mod), V[--i], mod);
        }
        partial_results[thread_id].value = sum;


        for (size_t step = 1; step < num_threads; step *= 2) {
            barrier.arrive_and_wait();
            if (thread_id % (2 * step) == 0 && thread_id + step < num_threads) {
                auto neighbor_range = vector_thread_range(N, num_threads, thread_id + step);
                partial_results[thread_id].value = add_mod(
                    partial_results[thread_id].value,
                    mul_mod(partial_results[thread_id + step].value,
                        word_pow_mod(neighbor_range.start - start, mod),
                        mod),
                    mod);
            }
        }
        };


    for (std::size_t i = 1; i < num_threads; ++i) {
        threads[i - 1] = std::thread(worker, i);
    }
    worker(0);

    for (auto& thread : threads) {
        thread.join();
    }
    return partial_results[0].value;
}
