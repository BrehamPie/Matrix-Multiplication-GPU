#include "gpu_tiled.h"

#define TILE_SIZE 32
#define TILE_M 2
#define TILE_N 2

__global__ void matmul_register_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
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

        // Load A - each thread loads multiple elements
        for (int i = 0; i < TILE_M; ++i) {
            for (int load_col = tx; load_col < TILE_SIZE; load_col += blockDim.x) {
                int row_idx = block_row + thread_row_start + i;
                int aCol = t + load_col;
                Asub[thread_row_start + i][load_col] = (row_idx < M && aCol < K) ? A[row_idx * K + aCol] : 0.0f;
            }
        }

        // Load B - each thread loads multiple elements
        for (int j = 0; j < TILE_N; ++j) {
            for (int load_row = ty; load_row < TILE_SIZE; load_row += blockDim.y) {
                int bRow = t + load_row;
                int col_idx = block_col + thread_col_start + j;
                Bsub[load_row][thread_col_start + j] = (bRow < K && col_idx < N) ? B[bRow * N + col_idx] : 0.0f;
            }
        }

        __syncthreads();

        // Compute partial results using registers
        for (int k = 0; k < TILE_SIZE; ++k) {
// Perform the multiply-accumulate directly
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

    // Write back results to global memory
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            int globalRow = block_row + thread_row_start + i;
            int globalCol = block_col + thread_col_start + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = regC[i][j];
            }
        }
    }
}

void gpu_matmul_register_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_register_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}