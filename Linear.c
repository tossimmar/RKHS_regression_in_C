#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "Linear.h"
#include "populate_matrix.h"
#include "populate_vector.h"

/*
Provides Matrix and Vector types and associated operations.
*/

// ------------------------------------------------------------------------------------------------

struct Matrix {
	/*
	Representation of a matrix. 

	attributes
	----------
	int : dim0
		The number of rows of the matrix.

	int : dim1
		The number of columns of the matrix.

	double ** : matrix
		The matrix with the specified dimensions.
	*/
	int dim0;
	int dim1;
	double **matrix;
};

// ------------------------------------------------------------------------------------------------

struct Vector {
	/*
	Representation of a Euclidean vector.

	attributes
	----------
	int : dim
		The dimension of the vector.

	double * : vector
		The vector with the specified dimension.
	*/
	int dim;
	double *vector;
};

// ------------------------------------------------------------------------------------------------

Matrix Matrix_new(int dim0, int dim1) {
	/*
	Construct a matrix of the specified dimensions.

	arguments
	---------
	int : dim0
		The number of rows of the matrix.

	int : dim1
		The number of columns of the matrix.

	returns
	-------
	Matrix
		An uninitialized matrix with the specified dimensions.
	*/
	Matrix m;
	m = malloc(sizeof(*m));
	assert(m);
	m->dim0 = dim0;
	m->dim1 = dim1;
	m->matrix = malloc(sizeof(double *) * dim0);
	assert(m->matrix);
	for(int i = 0; i < dim0; i++) {
		m->matrix[i] = malloc(sizeof(double) * dim1);
		assert(m->matrix[i]);
	}
	return m;
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_init(int dim0, int dim1, ...) {
	/*
	Initialize a matrix of the specified dimensions with the 
	specified elements.

	arguments
	---------
	int : dim0
		The number of rows of the matrix.

	int : dim1
		The number of columns of the matrix.

	...
		The elements of the matrix (which must be of type double).

	returns
	-------
	Matrix
		The matrix of the specified dimensions initialized with the 
		specified elements.

	example
	-------
	The call Matrix_init(2, 3, 1., 2., 3., 4., 5., 6.) returns the
	matrix

		[[1., 2., 3.], 
		 [4., 5., 6.]]
	*/
	va_list args;
	va_start(args, dim1);
	Matrix A = populate_matrix(Matrix_new(dim0, dim1), Matrix_init_populate, args);
	va_end(args);
	return A;
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_constant(int dim0, int dim1, double c) {
	/*
	Return a constant matrix of the specified dimensions.

	arguments
	---------
	int : dim0
		The number of rows of the matrix.

	int : dim1
		The number of columns of the matrix.

	double : c
		The value of every element of the matrix.

	returns
	-------
	Matrix
		The matrix of the specified dimensions with all 
		elements equal to the specified constant.

	example
	-------
	The call Matrix_constant(2, 3, 1.) returns the matrix

		[[1., 1., 1.], 
		 [1., 1., 1.]]
	*/
	return populate_matrix(Matrix_new(dim0, dim1), Matrix_constant_populate, c);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_identity(int dim) {
	/*
	Return the identity matrix of the specified dimension.

	arguments
	---------
	int : dim
		The number of rows and columns of the matrix.

	returns
	-------
	Matrix
		The identity matrix of the specified dimension.

	example
	-------
	The call Matrix_identity(3) returns the matrix

		[[1., 0., 0.], 
		 [0., 1., 0.], 
		 [0., 0., 1.]]
	*/
	return populate_matrix(Matrix_new(dim, dim), Matrix_identity_populate);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_scalar_mul(Matrix m, double c) {
	/*
	Scalar multiply the specified matrix by the specified constant.

	arguments
	---------
	Matrix : m
		The matrix to scalar multiply.

	double : c
		The scalar multiple.

	returns
	-------
	Matrix
		The specified matrix multiplied by the specified constant. 

	example
	-------
	Suppose that the specified matrix and constant are given by

		    [[1., 1.], 
		m =  [1., 2.],    and    c = 3. 
		     [3., 1.]]

	Then, the call Matrix_scalar_mul(m, c) returns the matrix

		[[3., 3.], 
		 [3., 6.], 
		 [9., 3.]] 
	*/
	return populate_matrix(Matrix_new(Matrix_dim(m, 0), Matrix_dim(m, 1)), Matrix_scalar_mul_populate, m, c);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_add(Matrix m1, Matrix m2) {
	/*
	Return the sum of the specified matrices.

	arguments
	---------
	Matrix : m1
		The first matrix in the sum.

	Matrix : m2
		The second matrix in the sum.

	returns
	-------
	Matrix
		The sum of the specified matrices.

	example
	-------
	Suppose that the specified matrices are given by

		     [[1., 0., 0.],		   [[0., 1., 1.],
		m1 =  [0., 1., 0.],    and    m2 =  [1., 0., 1.],
		      [0., 0., 1.]]                 [1., 1., 0.]]

	Then, the call Matrix_add(m1, m2) returns the matrix

		[[1., 1., 1.], 
		 [1., 1., 1.], 
		 [1., 1., 1.]]
	*/
	assert((Matrix_dim(m1, 0) == Matrix_dim(m2, 0)) && (Matrix_dim(m1, 1) == Matrix_dim(m2, 1)));
	return populate_matrix(Matrix_new(Matrix_dim(m1, 0), Matrix_dim(m1, 1)), Matrix_add_populate, m1, m2);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_sub(Matrix m1, Matrix m2) {
	/*
	Return the difference of the specified matrices.

	arguments
	---------
	Matrix : m1
		The first matrix in the difference.

	Matrix : m2
		The second matrix in the difference.

	returns
	-------
	Matrix
		The difference of the specified matrices.

	example
	-------
	Suppose that the specified matrices are given by

		     [[1., 0., 0.],		   [[0., 1., 1.],
		m1 =  [0., 1., 0.],    and    m2 =  [1., 0., 1.],
		      [0., 0., 1.]]                 [1., 1., 0.]]

	Then, the call Matrix_sub(m1, m2) returns the matrix

		[[ 1., -1., -1.], 
		 [-1.,  1., -1.], 
		 [-1., -1.,  1.]]
	*/
	assert((Matrix_dim(m1, 0) == Matrix_dim(m2, 0)) && (Matrix_dim(m1, 1) == Matrix_dim(m2, 1)));
	return populate_matrix(Matrix_new(Matrix_dim(m1, 0), Matrix_dim(m1, 1)), Matrix_sub_populate, m1, m2);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_hadamard(Matrix m1, Matrix m2) {
	/*
	Return the Hadamard product (i.e., element-wise product) 
	of the specified matrices.

	arguments
	---------
	Matrix : m1
		The first matrix in the product.

	Matrix : m2
		The second matrix in the product.

	returns
	-------
	Matrix
		The Hadamard product of the specified matrices.

	example
	-------
	Suppose that the specified matrices are given by

		     [[1., 2., 0.],		   [[0., 1., 1.],
		m1 =  [0., 1., 0.],    and    m2 =  [1., 0., 1.],
		      [3., 0., 2.]]                 [1., 1., 5.]]

	Then, the call Matrix_hadamard(m1, m2) returns the matrix

		[[0., 2.,  0.], 
		 [0., 0.,  0.], 
		 [3., 0., 10.]]
	*/
	assert((Matrix_dim(m1, 0) == Matrix_dim(m2, 0)) && (Matrix_dim(m1, 1) == Matrix_dim(m2, 1)));
	return populate_matrix(Matrix_new(Matrix_dim(m1, 0), Matrix_dim(m1, 1)), Matrix_hadamard_populate, m1, m2);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_matmul(Matrix m1, Matrix m2) {
	/*
	Return the matrix multiplication of the specified matrices.

	arguments
	---------
	Matrix : m1
		The first (i.e., left) matrix in the matrix multiplication.

	Matrix : m2
		The second (i.e., right) matrix in the matrix multiplication.

	returns
	-------
	Matrix
		The matrix multiplication of the specified matrices. 

	example
	-------
	Suppose that the specified matrices are given by

		     [[1., 0., 0.],		   [[0., 1., 2.],
		m1 =  [0., 1., 0.]]    and    m2 =  [1., 0., 1.],
						    [1., 1., 0.]]

	Then, the call Matrix_matmul(m1, m2) returns the matrix

		[[0., 1., 2.], 
		 [1., 0., 1.]] 
	*/
	assert(Matrix_dim(m1, 1) == Matrix_dim(m2, 0));
	return populate_matrix(Matrix_new(Matrix_dim(m1, 0), Matrix_dim(m2, 1)), Matrix_matmul_populate, m1, m2);
}

// ------------------------------------------------------------------------------------------------

Matrix Matrix_transpose(Matrix m) {
	/*
	Return the transpose of the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix to transpose.

	returns
	-------
	Matrix
		The transpose of the specified matrix.

	example
	-------
	Suppose that the specified matrix is given by

		    [[1., 2.], 
		m =  [3., 4.],
		     [5., 6.]]

	Then, the call Matrix_transpose(m) returns the matrix

		[[1., 3., 5.], 
		 [2., 4., 6.]]
	*/
	return populate_matrix(Matrix_new(Matrix_dim(m, 1), Matrix_dim(m, 0)), Matrix_transpose_populate, m);
}

// ------------------------------------------------------------------------------------------------

double Matrix_sum(Matrix m) {
	/*
	Return the sum of the elements of the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix to sum.

	returns
	-------
	double
		The sum of the elements of the specified matrix.

	example
	-------
	Suppose that the specified matrix is given by

		    [[1., 2.], 
		m =  [3., 4.],
		     [5., 6.]]

	Then, the call Matrix_sum(m) returns the scalar

		1. + 2. + 3. + 4. + 5. + 6. = 21.
	*/
	double sum = 0.0;
	for(int i = 0; i < Matrix_dim(m, 0); i++) {
		for(int j = 0; j < Matrix_dim(m, 1); j++) {
			sum += Matrix_get(m, i, j);
		}
	}
	return sum;
}

// ------------------------------------------------------------------------------------------------

double Matrix_frobenius_norm(Matrix m) {
	/*
	Return the Frobenius norm of the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix the Frobenius norm is with respect to.

	returns
	-------
	double
		The Frobenius norm of the specified matrix.

	example
	-------
	Suppose that the specified matrix is given by

		    [[1., 2.], 
		m =  [3., 4.]]

	Then, the call Matrix_frobenius_norm(m) returns the scalar

		sqrt(1.^2 + 2.^2 + 3.^2 + 4.^2) = sqrt(30.)
	*/
	Matrix prod = Matrix_hadamard(m, m);
	double norm = sqrt(Matrix_sum(prod));
	Matrix_delete(prod);
	return norm;
}

// ------------------------------------------------------------------------------------------------

double Matrix_get(Matrix m, int i, int j) {
	/*
	Return the element of the specified matrix with the specified indices.

	arguments
	---------
	Matrix : m
		The matrix to index.

	int : i
		The row index.

	int : j
		The column index.

	returns
	-------
	double
		The element of the specified matrix with the specified indices.

	example
	-------
	Suppose that the specified matrix and indices are given by

		    [[1.0, 2.0, 3.0], 
		m =  [4.0, 5.0, 6.0],    and    i = 1    and    j = 2
		     [7.0, 8.0, 9.0]]

	Then, the call Matrix_get(m, i, j) returns the scalar 6.0.
	*/
	assert((i >= 0 && i < Matrix_dim(m, 0)) && (j >= 0 && j < Matrix_dim(m, 1)));
	return m->matrix[i][j];
}

// ------------------------------------------------------------------------------------------------

Vector Matrix_get_row(Matrix m, int i) {
	/*
	Return the row of the specified matrix with the specified index.

	arguments
	---------
	Matrix : m
		The matrix to index.

	int : i
		The row index.

	returns
	-------
	Vector
		The row of the specified matrix with the specified index.

	example
	-------
	Suppose that the specified matrix and index are given by

		    [[1.0, 2.0, 3.0], 
		m =  [4.0, 5.0, 6.0],    and    i = 1
		     [7.0, 8.0, 9.0]]

	Then, the call Matrix_get_row(m, i) returns the vector

		[4.0, 5.0, 6.0]
	*/
	assert(i >= 0 && i < Matrix_dim(m, 0));
	Vector v = Vector_new(Matrix_dim(m, 1));
	memcpy(v->vector, m->matrix[i], sizeof(double) * Matrix_dim(m, 1));
	return v;
}

// ------------------------------------------------------------------------------------------------

void Matrix_set(Matrix m, int i, int j, double val) {
	/*
	Set the element of the specified matrix with the specified
	indices to the specified value.

	arguments
	---------
	Matrix : m
		The matrix to set.

	int : i
		The row index.

	int : j
		The column index.

	double : val
		The value to set.

	example
	-------
	Suppose that the specified matrix, indices, and value are given by

		    [[1.0, 2.0, 3.0], 
		m =  [4.0, 5.0, 6.0],    and    i = 1    and    val = 10.0
		     [7.0, 8.0, 9.0]]           j = 2

	Then, the call Matrix_set(m, i, j, val) changes the value 6.0 in m
	to the value 10.0.
	*/
	assert((i >= 0 && i < Matrix_dim(m, 0)) && (j >= 0 && j < Matrix_dim(m, 1)));
	m->matrix[i][j] = val;
}

// ------------------------------------------------------------------------------------------------

void Matrix_delete(Matrix m) {
	/*
	Delete the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix to delete.

	returns
	-------
	void
	*/
	for(int i = 0; i < Matrix_dim(m, 0); i++) {
		free(m->matrix[i]);
	}
	free(m->matrix);
	free(m);
}

// ------------------------------------------------------------------------------------------------

void Matrix_print(Matrix m) {
	/*
	Print the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix to print.

	returns
	-------
	void

	example
	-------
	Suppose that the specified matrix is given by

		    [[1.0, 2.0, 3.0], 
		m =  [4.0, 5.0, 6.0],
		     [7.0, 8.0, 9.0]]

	Then, the call Matrix_print(m) prints m as displayed above.
	*/
	for(int i = 0; i < Matrix_dim(m, 0); i++) {
		printf(i ? " [" : "[[");
		int j;
		for(j = 0; j < Matrix_dim(m, 1) - 1; j++) {
			printf("%.3f, ", Matrix_get(m, i, j));
		}
		printf("%.3f", Matrix_get(m, i, j));
		printf((i < Matrix_dim(m, 0) - 1) ? "]\n" : "]]\n");
	}
}

// ------------------------------------------------------------------------------------------------

int Matrix_dim(Matrix m, int dim) {
	/*
	Return the row/column dimension of the specified matrix.

	arguments
	---------
	Matrix : m
		The matrix the dimension is with respect to.

	int : dim
		If 0, return the number of rows.
		If 1, return the number of columns.

	returns
	-------
	int
		The row/column dimension of the specified matrix.

	example
	-------
	Suppose that the specified matrix is given by

		    [[1.0, 2.0, 3.0], 
		m =  [4.0, 5.0, 6.0]]

	Then, the calls Matrix_dim(m, 0) and Matrix_dim(m, 1) 
	return 2 and 3, respectively.
	*/
	assert(dim == 0 || dim == 1);
	return dim ? m->dim1 : m->dim0;
}

// ------------------------------------------------------------------------------------------------

Vector Vector_new(int dim) {
	Vector v;
	v = malloc(sizeof(*v));
	assert(v);
	v->dim = dim;
	v->vector = malloc(sizeof(double) * dim);
	assert(v->vector);
	return v;
}

// ------------------------------------------------------------------------------------------------

Vector Vector_init(int dim, ...) {
	va_list args;
	va_start(args, dim);
	Vector v = populate_vector(Vector_new(dim), Vector_init_populate, args);
	va_end(args);
	return v;
}

// ------------------------------------------------------------------------------------------------

Vector Vector_constant(int dim, double c) {
	return populate_vector(Vector_new(dim), Vector_constant_populate, c);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_scalar_mul(Vector v, double c) {
	return populate_vector(Vector_new(Vector_dim(v)), Vector_scalar_mul_populate, v, c);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_add(Vector v1, Vector v2) {
	assert(Vector_dim(v1) == Vector_dim(v2));
	return populate_vector(Vector_new(Vector_dim(v1)), Vector_add_populate, v1, v2);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_sub(Vector v1, Vector v2) {
	assert(Vector_dim(v1) == Vector_dim(v2));
	return populate_vector(Vector_new(Vector_dim(v1)), Vector_sub_populate, v1, v2);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_hadamard(Vector v1, Vector v2) {
	assert(Vector_dim(v1) == Vector_dim(v2));
	return populate_vector(Vector_new(Vector_dim(v1)), Vector_hadamard_populate, v1, v2);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_matmul(Matrix m, Vector v) {
	assert(Matrix_dim(m, 1) == Vector_dim(v));
	return populate_vector(Vector_new(Matrix_dim(m, 0)), Vector_matmul_populate, m, v);
}

// ------------------------------------------------------------------------------------------------

Vector Vector_uniform(double lo, double hi, int n, unsigned seed) {
	assert(lo < hi && n >= 1);
	srand(seed);
	return populate_vector(Vector_new(n), Vector_uniform_populate, lo, hi);
}

// ------------------------------------------------------------------------------------------------

double Vector_sum(Vector v) {
	double sum = 0.0;
	for(int i = 0; i < Vector_dim(v); i++) {
		sum += Vector_get(v, i);
	}
	return sum;
}

// ------------------------------------------------------------------------------------------------

double Vector_norm(Vector v) {
	return sqrt(Vector_inner(v, v));
}

// ------------------------------------------------------------------------------------------------

double Vector_inner(Vector v1, Vector v2) {
	assert(Vector_dim(v1) == Vector_dim(v2));
	Vector prod = Vector_hadamard(v1, v2);
	double inner = Vector_sum(prod);
	Vector_delete(prod);
	return inner;
}

// ------------------------------------------------------------------------------------------------

double Vector_get(Vector v, int i) {
	assert(i >= 0 && i < Vector_dim(v));
	return v->vector[i];
}

// ------------------------------------------------------------------------------------------------

int Vector_dim(Vector v) {
	return v->dim;
}

// ------------------------------------------------------------------------------------------------

void Vector_delete(Vector v) {
	free(v->vector);
	free(v);
}

// ------------------------------------------------------------------------------------------------

void Vector_set(Vector v, int i, double val) {
	assert(i >= 0 && i < Vector_dim(v));
	v->vector[i] = val;
}

// ------------------------------------------------------------------------------------------------

void Vector_print(Vector v) {
	printf("[");
	int i;
	for(i = 0; i < Vector_dim(v) - 1; i++) {
		printf("%.3f, ", Vector_get(v, i));
	}
	printf("%.3f]\n", Vector_get(v, i));
}
