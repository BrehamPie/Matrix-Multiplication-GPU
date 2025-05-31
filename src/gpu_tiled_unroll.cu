#include "gpu_tiled_unroll.h"

#define TILE_SIZE 64
#define TILE_M 4
#define TILE_N 4

__global__
void matmul_register_kernel_float4(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate block base row and col in C
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    // Thread's position within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate which elements this thread will compute
    int thread_row_start = ty * TILE_M;
    int thread_col_start = tx * TILE_N;

    // Register tile to accumulate partial results
    float regC[TILE_M][TILE_N] = {0.0f};
    
    // Loop over tiles of A and B along the K dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        __shared__ float Asub[TILE_SIZE][TILE_SIZE];
        __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

        // Load A using float4 - need to ensure complete coverage
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int row_idx = block_row + thread_row_start + i;
            
            // Each thread loads multiple sets of 4 elements to cover the full row
            for (int load_col = tx * 4; load_col < TILE_SIZE; load_col += blockDim.x * 4) {
                int aCol = t + load_col;
                
                if (row_idx < M && aCol + 3 < K) {
                    // Use float4 for aligned loads
                    float4 a_vals = *reinterpret_cast<const float4*>(&A[row_idx * K + aCol]);
                    Asub[thread_row_start + i][load_col] = a_vals.x;
                    Asub[thread_row_start + i][load_col + 1] = a_vals.y;
                    Asub[thread_row_start + i][load_col + 2] = a_vals.z;
                    Asub[thread_row_start + i][load_col + 3] = a_vals.w;
                } else if (row_idx < M) {
                    // Handle boundary - load remaining elements individually
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        if (aCol + k < K) {
                            Asub[thread_row_start + i][load_col + k] = A[row_idx * K + aCol + k];
                        } else {
                            Asub[thread_row_start + i][load_col + k] = 0.0f;
                        }
                    }
                } else {
                    // Row out of bounds - zero out
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        Asub[thread_row_start + i][load_col + k] = 0.0f;
                    }
                }
            }
        }

        // Load B - need to ensure complete coverage
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int col_idx = block_col + thread_col_start + j;
            
            // Each thread loads multiple rows to cover the full column
            for (int load_row = ty; load_row < TILE_SIZE; load_row += blockDim.y) {
                int bRow = t + load_row;
                Bsub[load_row][thread_col_start + j] = 
                    (bRow < K && col_idx < N) ? B[bRow * N + col_idx] : 0.0f;
            }
        }

        __syncthreads();

        // Compute partial results
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < TILE_N; ++j) {
                    regC[i][j] += Asub[thread_row_start + i][k] * Bsub[k][thread_col_start + j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results
    #pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int globalRow = block_row + thread_row_start + i;
            int globalCol = block_col + thread_col_start + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = regC[i][j];
            }
        }
    }
}
#include <stdio.h>
void gpu_matmul_register_tiled_vectorized(const float *A, const float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Keep original block dimensions but use float4 in loading
    dim3 blockDim(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);  // Back to (16, 16)
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_register_kernel_float4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}