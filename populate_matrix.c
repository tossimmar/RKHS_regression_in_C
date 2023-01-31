#include <stdarg.h>
#include "Linear.h"
#include "RBF.h"

Matrix populate_matrix(Matrix A, double (*fp)(int, int, va_list), ...) {
	va_list args;
	for(int i = 0; i < Matrix_dim(A, 0); i++) {
		for(int j = 0; j < Matrix_dim(A, 1); j++) {
			va_start(args, fp);
			Matrix_set(A, i, j, fp(i, j, args));
			va_end(args);
		}
	}
	return A;
}

double Matrix_init_populate(int i, int j, va_list args) {
	va_list *elements = va_arg(args, va_list *);
	return va_arg(elements, double);
}

double Matrix_constant_populate(int i, int j, va_list args) {
	return va_arg(args, double);
}

double Matrix_identity_populate(int i, int j, va_list args) {
	return (i == j) ? 1.0 : 0.0;
}

double Matrix_scalar_mul_populate(int i, int j, va_list args) {
	Matrix m = va_arg(args, Matrix);
	double c = va_arg(args, double);
	return c * Matrix_get(m, i, j);
}

double Matrix_add_populate(int i, int j, va_list args) {
	Matrix m1 = va_arg(args, Matrix);
	Matrix m2 = va_arg(args, Matrix);
	return Matrix_get(m1, i, j) + Matrix_get(m2, i, j);
}

double Matrix_sub_populate(int i, int j, va_list args) {
	Matrix m1 = va_arg(args, Matrix);
	Matrix m2 = va_arg(args, Matrix);
	return Matrix_get(m1, i, j) - Matrix_get(m2, i, j);
}

double Matrix_hadamard_populate(int i, int j, va_list args) {
	Matrix m1 = va_arg(args, Matrix);
	Matrix m2 = va_arg(args, Matrix);
	return Matrix_get(m1, i, j) * Matrix_get(m2, i, j);
}

double Matrix_matmul_populate(int i, int j, va_list args) {
	Matrix m1 = va_arg(args, Matrix);
	Matrix m2 = va_arg(args, Matrix);
	double out = 0.0;
	for(int k = 0; k < Matrix_dim(m1, 1); k++) {
		out += Matrix_get(m1, i, k) * Matrix_get(m2, k, j);
	}
	return out;
}

double Matrix_transpose_populate(int i, int j, va_list args) {
	Matrix m = va_arg(args, Matrix);
	return Matrix_get(m, j, i);
}

double RBF_kernel_matrix_populate(int i, int j, va_list args) {
	Matrix X1 = va_arg(args, Matrix);
	Matrix X2 = va_arg(args, Matrix);
	double s = va_arg(args, double);
	Vector x1 = Matrix_get_row(X1, i);
	Vector x2 = Matrix_get_row(X2, j);
	double out = RBF_kernel(x1, x2, s);
	Vector_delete(x1);
	Vector_delete(x2);
	return out;
}