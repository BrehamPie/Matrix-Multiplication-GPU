#ifndef GPU_SHARED_H
#define GPU_SHARED_H

void gpu_matmul_shared(const float* A, const float* B, float* C, int M, int N, int K);
#endif