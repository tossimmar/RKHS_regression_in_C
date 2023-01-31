#ifndef LINEAR_INCLUDED
#define LINEAR_INCLUDED

typedef struct Matrix * Matrix;
typedef struct Vector * Vector;

Matrix Matrix_new(int, int);
Matrix Matrix_init(int, int, ...);
Matrix Matrix_constant(int, int, double);
Matrix Matrix_identity(int);
Matrix Matrix_scalar_mul(Matrix, double);
Matrix Matrix_add(Matrix, Matrix);
Matrix Matrix_sub(Matrix, Matrix);
Matrix Matrix_hadamard(Matrix, Matrix);
Matrix Matrix_matmul(Matrix, Matrix);
Matrix Matrix_transpose(Matrix);
Vector Matrix_get_row(Matrix, int);
double Matrix_sum(Matrix);
double Matrix_frobenius_norm(Matrix);
double Matrix_get(Matrix, int, int);
int Matrix_dim(Matrix, int);
void Matrix_set(Matrix, int, int, double);
void Matrix_delete(Matrix);
void Matrix_print(Matrix);

Vector Vector_new(int);
Vector Vector_init(int, ...);
Vector Vector_constant(int, double);
Vector Vector_scalar_mul(Vector, double);
Vector Vector_add(Vector, Vector);
Vector Vector_sub(Vector, Vector);
Vector Vector_hadamard(Vector, Vector);
Vector Vector_matmul(Matrix, Vector);
Vector Vector_uniform(double, double, int, unsigned);
double Vector_sum(Vector);
double Vector_norm(Vector);
double Vector_inner(Vector, Vector);
double Vector_get(Vector, int);
int Vector_dim(Vector);
void Vector_set(Vector, int, double);
void Vector_delete(Vector);
void Vector_print(Vector);

#endif