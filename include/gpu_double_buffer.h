#pragma once
void gpu_matmul_register_tiled_double_buffered(const float *A, const float *B, float *C, int M, int N, int K);