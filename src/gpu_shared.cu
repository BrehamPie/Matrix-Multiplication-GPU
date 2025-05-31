#include "gpu_shared.h"

#define TILE_SIZE 32

//Let's understand what's happening here:
// Our matrix of size N*N is broken into multiple submatrices of size TILE_SIZE*TILE_SIZE.
// Each thread block will compute one of these submatrices.
// Each thread is handling one element of the output matrix C.

// In naive eah thread reads N + N global memory locations.
// Total threads N*N
// Total reads N*N*(N+N) = 2*N^3

// Divide matrix into tiles of size T * T
// Each thread block: loads a tile of A and B into shared memory.
// Then does T iterations of multiply-add using the tile.
// Each tile is loaded once per block, and shared among T * T threads.
// Assume N % T = 0.
// Blocks per dim: N / T
// Total blocks: (N/T) * (N/T) = (N/T)^2
// Each block loads T*T elements of A and B.
// And does this N / T times.
// Total reads: (N/T) * T^2 * 2 = 2*NT per block.
// Total reads: (N/T)^2 * 2*NT = 2*N^3 / T.
// so if T = 32, then we get 32x speedup over naive method.


// In shared memory each thread block reads TILE_SIZE*TILE_SIZE global memory locations.
// __shared__ memory is shared among threads in the same block.
// normal variable is private to each thread.
// We are actually loading the whole matrix A and B, but earlier we were loading it in each thread.
// Now we are loading it once per block, and then each thread in the block can access it.
__global__
void matmul_shared_kernel(const float* A, const float* B, float* C, int M,int N, int K){
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f; // for each thread.

    for(int t = 0; t< (K+TILE_SIZE - 1) / TILE_SIZE; t++){
        // Load tile from A and B into shared memory
        if( row<M and t* TILE_SIZE + threadIdx.x <K){
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
            // row*K th row, t*tile_size start of column, x offset
        } else{
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N and t* TILE_SIZE + threadIdx.y < K){
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }else{
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Synchronize to ensure all threads have loaded their tiles
        // 32*32 size of tile_A and tile_B is loaded into shared memory
        __syncthreads();
        // Perform the multiplication for this tile
        // Each element of C gets a small contribution from each tile.
        for(int i = 0; i <TILE_SIZE;i++){
            sum+= tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if( row <M and col < N){
        C[row * N + col] = sum;
    }
}

void gpu_matmul_shared(const float* A, const float* B, float* C, int M, int N, int K){
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}