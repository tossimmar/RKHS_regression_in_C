#include <math.h>
#include "Linear.h"
#include "populate_matrix.h"

double RBF_kernel(Vector x1, Vector x2, double s) {
	/*
	Return the evaluation of the RBF kernel with the specified 
	parameter at the specified vectors. 

	arguments 
	---------
	Vector : x1
		The first argument of the RBF kernel.

	Vector : x2
		The second argument of the RBF kernel.

	double : s
		The RBF kernel parameter.

	returns
	-------
	double
		The RBF kernel evaluated at the specified vectors. 
	*/
	Vector diff = Vector_sub(x1, x2);
	double norm2 = Vector_inner(diff, diff);
	double coef = -(1 / (2 * pow(s, 2)));
	double out = exp(coef * norm2);
	Vector_delete(diff);
	return out;
}

Matrix RBF_kernel_matrix(Matrix X1, Matrix X2, double s) {
	/*
	Return the RBF kernel matrix.

	arguments
	---------
	Matrix : X1
		The matrix which determines the row dimension of
		the kernel matrix (e.g., a test data matrix).

	Matrix : X2
		The matrix which determines the column dimension of
		the kernel matrix (e.g., a training data matrix).

	double : s
		The RBF kernel parameter. If the second data matrix is
		a training data matrix, then this argument should be
		equal to the RBF kernel parameter used in training.

	returns
	-------
	Matrix
		The RBF kernel matrix.
	*/
	return populate_matrix(Matrix_new(Matrix_dim(X1, 0), Matrix_dim(X2, 0)), RBF_kernel_matrix_populate, X1, X2, s);
}

Vector RBF_fit(Matrix X, Vector y, double s, double r, double lr, int n) {
	/*
	Fit an RBF kernel regression model to the specified data.

	The loss function is given by

		L(a) = || y - Ka ||22 + r(aT)Ka

	where a is the vector of model parameters to be learned, ||.||22 denotes the 
	squared 2-norm, y is the specified target vector, K is the RBF kernel matrix 
	with respect to the specified data matrix and the RBF kernel parameter, r is 
	a regularization tuning parameter, and T denotes transpose. 

	The gradient of the loss function is given by

		dL/da = -2K(y - (K + rI)a)

	The function f (i.e., the model) associated with a solution a* = argmin L(a) 
	is given by

		f(x) = a1 * k(x, x1) + a2 * k(x, x2) + ... + aN * k(x, xN)

	where a* = (a1, ..., aN), k is the RBF kernel, and x1, ..., xN is the training
	data (i.e., the rows of the specified data matrix).

	arguments
	---------
	Matrix : X
		The feature matrix.

	Vector : y
		The target vector.

	double : s
		The RBF kernel parameter.

	double : r
		The regularization parameter.

	double : lr
		The learning rate used in gradient descent.

	int : n
		The number of gradient descent iterations.

	returns
	-------
	Vector
		The vector of learned model parameters (i.e., function coefficients). 
	*/
	Matrix K = RBF_kernel_matrix(X, X, s);
	Vector coef = Vector_uniform(-2, 2, Matrix_dim(X, 0), 1); // initialize model parameters
	Matrix I = Matrix_identity(Matrix_dim(X, 0));
	Matrix rI = Matrix_scalar_mul(I, r);
	Matrix KrI = Matrix_add(K, rI);
	for(int i = 0; i < n; i++) {
		Vector coef_old = coef;
		Vector v1 = Vector_matmul(KrI, coef);
		Vector v2 = Vector_sub(y, v1);
		Vector v3 = Vector_matmul(K, v2);
		Vector lr_grad = Vector_scalar_mul(v3, -2 * lr); // learning rate * gradient
		coef = Vector_sub(coef_old, lr_grad); // gradient descent update
		Vector_delete(coef_old);
		Vector_delete(v1);
		Vector_delete(v2);
		Vector_delete(v3);
		Vector_delete(lr_grad);
	}
	Matrix_delete(K);
	Matrix_delete(I);
	Matrix_delete(rI);
	Matrix_delete(KrI);
	return coef;
}

Vector RBF_predict(Matrix X_test, Matrix X_train, Vector coef, double s) {
	/*
	Return a vector of predictions from the trained RBF kernel 
	regression model with the specified parameters.

	arguments
	---------
	Matrix : X_test
		The test feature matrix.

	Matrix : X_train
		The training feature matrix.

	Vector : coef
		The trained model parameters (i.e., function coefficients).

	double : s
		The RBF kernel parameter used in training.

	returns
	-------
	Vector
		The vector of predictions.
	*/
	Matrix K = RBF_kernel_matrix(X_test, X_train, s);
	Vector predictions = Vector_matmul(K, coef);
	Matrix_delete(K);
	return predictions;
}