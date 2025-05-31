#include "utils.h"

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    }
}
bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols, float eps) {
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(mat1[i] - mat2[i]) > eps) {
          //  printf("Difference at index %d: %f vs %f\n", i, mat1[i], mat2[i]);
            return false;
        }
    }
    return true;
}

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}