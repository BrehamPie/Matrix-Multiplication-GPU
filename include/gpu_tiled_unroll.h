#pragma once
void gpu_matmul_register_tiled_vectorized(const float* A, const float* B, float* C, int M, int N, int K);