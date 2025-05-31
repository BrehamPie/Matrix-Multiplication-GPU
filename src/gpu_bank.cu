#include <cuda_runtime.h>
#include "gpu_warp.h"

#define TILE_SIZE 32
#define TILE_M 2
#define TILE_N 2

__global__
void gpu_matmul_register_tiled_bank_conflict_free_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int thread_row_start = ty * TILE_M;
    int thread_col_start = tx * TILE_N;

    float regC[TILE_M][TILE_N] = {0.0f};

    for (int t = 0; t < K; t += TILE_SIZE) {
        __shared__ float Asub[TILE_SIZE][TILE_SIZE+1];
        __shared__ float Bsub[TILE_SIZE][TILE_SIZE+1];

        // Load A - need to ensure full 64x64 coverage
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int row_idx = block_row + thread_row_start + i;
            // Each thread loads multiple sets of 4 to cover full tile
            for (int load_col = tx * 4; load_col < TILE_SIZE; load_col += blockDim.x * 4) {
                int aCol = t + load_col;
                if (row_idx < M && aCol + 3 < K) {
                    float4 a_vals = *reinterpret_cast<const float4*>(&A[row_idx * K + aCol]);
                    Asub[thread_row_start + i][load_col] = a_vals.x;
                    Asub[thread_row_start + i][load_col + 1] = a_vals.y;
                    Asub[thread_row_start + i][load_col + 2] = a_vals.z;
                    Asub[thread_row_start + i][load_col + 3] = a_vals.w;
                } else if (row_idx < M) {
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        if (aCol + k < K) {
                            Asub[thread_row_start + i][load_col + k] = A[row_idx * K + aCol + k];
                        } else {
                            Asub[thread_row_start + i][load_col + k] = 0.0f;
                        }
                    }
                } else {
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        Asub[thread_row_start + i][load_col + k] = 0.0f;
                    }
                }
            }
        }

        // Load B - need to ensure full 64x64 coverage
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int col_idx = block_col + thread_col_start + j;
            // Each thread loads multiple rows to cover full tile
            for (int load_row = ty; load_row < TILE_SIZE; load_row += blockDim.y) {
                int bRow = t + load_row;
                Bsub[load_row][thread_col_start + j] = 
                    (bRow < K && col_idx < N) ? B[bRow * N + col_idx] : 0.0f;
            }
        }

        __syncthreads();

        // Compute - this part was actually correct
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val[TILE_M];
            #pragma unroll
            for (int i = 0; i < TILE_M; ++i) {
                a_val[i] = Asub[thread_row_start + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                float b_val = Bsub[k][thread_col_start + j];
                #pragma unroll
                for (int i = 0; i < TILE_M; ++i) {
                    regC[i][j] += a_val[i] * b_val;
                }
            }
        }

        __syncthreads();
    }

    // Write back results - REMOVE THE BROKEN WARP SHUFFLE
    #pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int globalRow = block_row + thread_row_start + i;
            int globalCol = block_col + thread_col_start + j;
            if (globalRow < M && globalCol < N) {
                // Just write the correct value - no shuffle needed here!
                C[globalRow * N + globalCol] = regC[i][j];
            }
        }
    }
}

void gpu_matmul_register_tiled_bank_conflict_free(const float *A, const float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);  // (16, 16)
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gpu_matmul_register_tiled_bank_conflict_free_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}