
#include "cpu_matmul.h"
#include "gpu_bank.h"
#include "gpu_double_buffer.h"
#include "gpu_naive.h"
#include "gpu_shared.h"
#include "gpu_shared_unroll.h"
#include "gpu_tiled.h"
#include "gpu_tiled_unroll.h"
#include "utils.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;
// const int N = 1024;
//  Number of iterations for timing
constexpr int ITER = 5;
float time_gpu_kernel(void (*kernel)(const float *, const float *, float *, int, int, int),
                      float *a, float *b, float *c, int M, int K, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0.0f;

    for (int i = 0; i < ITER; ++i) {
        cudaMemset(c, 0, M * N * sizeof(float));
        cudaEventRecord(start);
        kernel(a, b, c, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / ITER;
}

int main() {

    for (int N = 16; N < 10000; N *= 2) {
        size_t bytes = N * N * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c_cpu = (float *)malloc(bytes);
        float *h_c_gpu = (float *)malloc(bytes);
        float *h_c_reg = (float *)malloc(bytes);

        init_matrix(h_a, N, N);
        init_matrix(h_b, N, N);
        // CPU Timing
        auto start_cpu = chrono::high_resolution_clock::now();
        cpu_matmul(h_a, h_b, h_c_cpu, N, N, N);
        auto end_cpu = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
        // cout << "CPU Time (1 run): " << duration_cpu.count() << " ms\n\n";

        // Warm up GPU
        cudaFree(0);

        // cout << "Averaged GPU Times over " << ITER << " runs:\n";

        float t_naive = time_gpu_kernel(gpu_matmul_naive, h_a, h_b, h_c_gpu, N, N, N);
        // cout << "GPU Naive: " << t_naive << " ms\n";

        float t_shared = time_gpu_kernel(gpu_matmul_shared, h_a, h_b, h_c_gpu, N, N, N);
        // cout << "GPU Shared: " << t_shared << " ms\n";

        float t_unroll = time_gpu_kernel(gpu_matmul_shared_unroll, h_a, h_b, h_c_gpu, N, N, N);
        // cout << "GPU Shared (Unroll): " << t_unroll << " ms\n";

        float t_register = time_gpu_kernel(gpu_matmul_register_tiled, h_a, h_b, h_c_reg, N, N, N);
        // cout << "GPU Register Tiled: " << t_register << " ms\n";

        float t_vector = time_gpu_kernel(gpu_matmul_register_tiled_vectorized, h_a, h_b, h_c_reg, N, N, N);
        // cout << "GPU Register Tiled (Vectorized): " << t_vector << " ms\n";

        float t_double = time_gpu_kernel(gpu_matmul_register_tiled_double_buffered, h_a, h_b, h_c_reg, N, N, N);
        // cout << "GPU Register Tiled (Double Buffer): " << t_double << " ms\n";

        float t_conflict_free = time_gpu_kernel(gpu_matmul_register_tiled_bank_conflict_free, h_a, h_b, h_c_reg, N, N, N);
        // cout << "GPU Register Tiled (Bank Conflict Free): " << t_conflict_free << " ms\n";
        cout << N << "," << duration_cpu.count() << ","
             << t_naive << "," << t_shared << "," << t_unroll << ","
             << t_register << "," << t_vector << ","
             << t_double << "," << t_conflict_free << "\n"; // Cleanup
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
        free(h_c_reg);
    }

    return 0;
}
