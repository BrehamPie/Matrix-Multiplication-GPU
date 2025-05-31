#include "cpu_matmul.h"
#include <omp.h>
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 64;  // Tune this for your CPU cache size

    // Initialize C to zero
    #pragma omp parallel for
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    // Blocked matrix multiplication with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {

                int i_max = std::min(i0 + BLOCK_SIZE, M);
                int j_max = std::min(j0 + BLOCK_SIZE, N);
                int k_max = std::min(k0 + BLOCK_SIZE, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int k = k0; k < k_max; ++k) {
                        float a_val = A[i * K + k];
                        for (int j = j0; j < j_max; ++j) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}