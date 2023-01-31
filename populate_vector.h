#ifndef POPULATE_VECTOR_INCLUDED
#define POPULATE_VECTOR_INCLUDED

#include <stdarg.h>
#include "Linear.h"

Vector populate_vector(Vector, double (*)(int, va_list), ...);
double Vector_init_populate(int, va_list);
double Vector_constant_populate(int, va_list);
double Vector_scalar_mul_populate(int, va_list);
double Vector_add_populate(int, va_list);
double Vector_sub_populate(int, va_list);
double Vector_hadamard_populate(int, va_list);
double Vector_matmul_populate(int, va_list);
double Vector_uniform_populate(int, va_list);

#endif