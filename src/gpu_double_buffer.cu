#include "gpu_double_buffer.h"

#define TILE_SIZE 32
#define TILE_M 2
#define TILE_N 2
// Won't work if already compute-bounded.
__global__
void matmul_register_kernel_double_buffered(const float* A, const float* B, float* C, int M, int N, int K) {
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
    
    // Double buffered shared memory - 2x the memory!
    __shared__ float Asub[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[2][TILE_SIZE][TILE_SIZE];
    
    // Buffer management
    int write_stage = 0;
    int read_stage = 1;
    
    // Pre-load the first tile into buffer 0
    int t = 0;
    {
        // Load A using float4
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int row_idx = block_row + thread_row_start + i;
            
            for (int load_col = tx * 4; load_col < TILE_SIZE; load_col += blockDim.x * 4) {
                int aCol = t + load_col;
                
                if (row_idx < M && aCol + 3 < K) {
                    float4 a_vals = *reinterpret_cast<const float4*>(&A[row_idx * K + aCol]);
                    Asub[write_stage][thread_row_start + i][load_col] = a_vals.x;
                    Asub[write_stage][thread_row_start + i][load_col + 1] = a_vals.y;
                    Asub[write_stage][thread_row_start + i][load_col + 2] = a_vals.z;
                    Asub[write_stage][thread_row_start + i][load_col + 3] = a_vals.w;
                } else if (row_idx < M) {
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        if (aCol + k < K) {
                            Asub[write_stage][thread_row_start + i][load_col + k] = A[row_idx * K + aCol + k];
                        } else {
                            Asub[write_stage][thread_row_start + i][load_col + k] = 0.0f;
                        }
                    }
                } else {
                    for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                        Asub[write_stage][thread_row_start + i][load_col + k] = 0.0f;
                    }
                }
            }
        }

        // Load B
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int col_idx = block_col + thread_col_start + j;
            
            for (int load_row = ty; load_row < TILE_SIZE; load_row += blockDim.y) {
                int bRow = t + load_row;
                Bsub[write_stage][load_row][thread_col_start + j] = 
                    (bRow < K && col_idx < N) ? B[bRow * N + col_idx] : 0.0f;
            }
        }
        
        __syncthreads();
    }
    
    // Main pipeline loop
    for (t = TILE_SIZE; t <= K; t += TILE_SIZE) {
        // Swap buffers for next iteration
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
        
        // Start loading the next tile (if it exists) while computing current tile
        if (t < K) {
            // Load A using float4 into write_stage buffer
            #pragma unroll
            for (int i = 0; i < TILE_M; ++i) {
                int row_idx = block_row + thread_row_start + i;
                
                for (int load_col = tx * 4; load_col < TILE_SIZE; load_col += blockDim.x * 4) {
                    int aCol = t + load_col;
                    
                    if (row_idx < M && aCol + 3 < K) {
                        float4 a_vals = *reinterpret_cast<const float4*>(&A[row_idx * K + aCol]);
                        Asub[write_stage][thread_row_start + i][load_col] = a_vals.x;
                        Asub[write_stage][thread_row_start + i][load_col + 1] = a_vals.y;
                        Asub[write_stage][thread_row_start + i][load_col + 2] = a_vals.z;
                        Asub[write_stage][thread_row_start + i][load_col + 3] = a_vals.w;
                    } else if (row_idx < M) {
                        for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                            if (aCol + k < K) {
                                Asub[write_stage][thread_row_start + i][load_col + k] = A[row_idx * K + aCol + k];
                            } else {
                                Asub[write_stage][thread_row_start + i][load_col + k] = 0.0f;
                            }
                        }
                    } else {
                        for (int k = 0; k < 4 && load_col + k < TILE_SIZE; ++k) {
                            Asub[write_stage][thread_row_start + i][load_col + k] = 0.0f;
                        }
                    }
                }
            }

            // Load B into write_stage buffer
            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                int col_idx = block_col + thread_col_start + j;
                
                for (int load_row = ty; load_row < TILE_SIZE; load_row += blockDim.y) {
                    int bRow = t + load_row;
                    Bsub[write_stage][load_row][thread_col_start + j] = 
                        (bRow < K && col_idx < N) ? B[bRow * N + col_idx] : 0.0f;
                }
            }
        }
        
        // Compute using the read_stage buffer (previous iteration's data)
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < TILE_N; ++j) {
                    regC[i][j] += Asub[read_stage][thread_row_start + i][k] * 
                                  Bsub[read_stage][k][thread_col_start + j];
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

void gpu_matmul_register_tiled_double_buffered(const float *A, const float *B, float *C, int M, int N, int K) {
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

    matmul_register_kernel_double_buffered<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}