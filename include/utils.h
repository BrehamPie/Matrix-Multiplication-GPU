#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>

void init_matrix(float* mat, int rows, int cols);
bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols, float eps = 1e-4f);
void print_matrix(const float* mat, int rows, int cols);
#endif