#ifndef GPU_SHARED_U_H
#define GPU_SHARED_U_H

void gpu_matmul_shared_unroll(const float* A, const float* B, float* C, int M, int N, int K);
#endif