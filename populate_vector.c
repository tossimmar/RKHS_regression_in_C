#include <stdlib.h>
#include <stdarg.h>
#include "Linear.h"

Vector populate_vector(Vector v, double (*fp)(int, va_list), ...) {
	va_list args;
	for(int i = 0; i < Vector_dim(v); i++) {
		va_start(args, fp);
		Vector_set(v, i, fp(i, args));
		va_end(args);
	}
	return v;
}

double Vector_init_populate(int i, va_list args) {
	va_list *elements = va_arg(args, va_list *);
	return va_arg(elements, double);
}

double Vector_constant_populate(int i, va_list args) {
	return va_arg(args, double);
}

double Vector_scalar_mul_populate(int i, va_list args) {
	Vector v = va_arg(args, Vector);
	double c = va_arg(args, double);
	return c * Vector_get(v, i);
}

double Vector_add_populate(int i, va_list args) {
	Vector v1 = va_arg(args, Vector);
	Vector v2 = va_arg(args, Vector);
	return Vector_get(v1, i) + Vector_get(v2, i);
}

double Vector_sub_populate(int i, va_list args) {
	Vector v1 = va_arg(args, Vector);
	Vector v2 = va_arg(args, Vector);
	return Vector_get(v1, i) - Vector_get(v2, i);
}

double Vector_hadamard_populate(int i, va_list args) {
	Vector v1 = va_arg(args, Vector);
	Vector v2 = va_arg(args, Vector);
	return Vector_get(v1, i) * Vector_get(v2, i);
}

double Vector_matmul_populate(int i, va_list args) {
	Matrix m = va_arg(args, Matrix);
	Vector v = va_arg(args, Vector);
	double out = 0.0;
	for(int j = 0; j < Matrix_dim(m, 1); j++) {
		out += Matrix_get(m, i, j) * Vector_get(v, j);
	}
	return out;
}

double Vector_uniform_populate(int i, va_list args) {
	double lo = va_arg(args, double);
	double hi = va_arg(args, double);
	double val = rand();
	double max = RAND_MAX;
	return lo + ((val / max) * (hi - lo));
}