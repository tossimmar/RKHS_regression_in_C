#ifndef RBF_INCLUDED
#define RBF_INCLUDED

#include "Linear.h"

double RBF_kernel(Vector, Vector, double);
Matrix RBF_kernel_matrix(Matrix, Matrix, double);
Vector RBF_fit(Matrix, Vector, double, double, double, int);
Vector RBF_predict(Matrix, Matrix, Vector, double);

#endif