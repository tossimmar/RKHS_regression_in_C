#ifndef POPULATE_MATRIX_INCLUDED
#define POPULATE_MATRIX_INCLUDED

#include <stdarg.h>
#include "Linear.h"

Matrix populate_matrix(Matrix, double (*)(int, int, va_list), ...);
double Matrix_init_populate(int, int, va_list);
double Matrix_constant_populate(int, int, va_list);
double Matrix_identity_populate(int, int, va_list);
double Matrix_scalar_mul_populate(int, int, va_list);
double Matrix_add_populate(int, int, va_list);
double Matrix_sub_populate(int, int, va_list);
double Matrix_hadamard_populate(int, int, va_list);
double Matrix_matmul_populate(int, int, va_list);
double Matrix_transpose_populate(int, int, va_list);
double RBF_kernel_matrix_populate(int, int, va_list);

#endif