#include "gpu_naive.h"


__global__
void matmul_naive_kernel(float* A, float* B, float* C, int M, int N,int K){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<M and col<N){
        float sum = 0.0f;
        for(int k=0;k<K;k++){
            sum+=A[row*K+k] * B[k*N+col];
        }
        C[row*N+col] = sum;
    }
}
void gpu_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K){
    // allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((N+15)/16, (M+15)/16);

    matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}