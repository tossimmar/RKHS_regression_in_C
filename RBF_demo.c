#include <stdio.h>
#include "RBF.h"
#include "Linear.h"

int main(int argc, char *argv[]) {
	/*
	Fit a 1-dimensional reproducing kernel Hilbert space (RKHS) 
	regression model. 

	The true relationship is f(x) = cos(5x) + sin(3x). The target
	vector below is given by yi = f(xi) + ni where the xi are 50
	evenly spaced points between 0 and 2pi and the ni are samples 
	from a Gaussian with mean 0 and standard deviation 0.4.
	*/

	// number of data points
	int N_SAMPLES = 50;

	// feature matrix
	Matrix X = Matrix_init(N_SAMPLES, 1, 0.        , 0.12822827, 0.25645654, 0.38468481, 0.51291309,
										 0.64114136, 0.76936963, 0.8975979 , 1.02582617, 1.15405444,
										 1.28228272, 1.41051099, 1.53873926, 1.66696753, 1.7951958 ,
										 1.92342407, 2.05165235, 2.17988062, 2.30810889, 2.43633716,
										 2.56456543, 2.6927937 , 2.82102197, 2.94925025, 3.07747852,
										 3.20570679, 3.33393506, 3.46216333, 3.5903916 , 3.71861988,
										 3.84684815, 3.97507642, 4.10330469, 4.23153296, 4.35976123,
										 4.48798951, 4.61621778, 4.74444605, 4.87267432, 5.00090259,
										 5.12913086, 5.25735913, 5.38558741, 5.51381568, 5.64204395,
										 5.77027222, 5.89850049, 6.02672876, 6.15495704, 6.28318531);

	// target vector
	Vector y = Vector_init(N_SAMPLES, 0.47825676,  1.398221  ,  0.23357338,  0.42024866,  0.25953002,
									  0.14530435, -0.50660754, -0.13471363,  0.48792933,  0.51144912,
									  0.11560712, -0.19524861, -0.82276978, -1.28726458, -1.40249529,
									 -1.51942015, -1.08053275,  0.23935084,  0.8847282 ,  1.40290518,
									  2.4786948 ,  1.79443347,  0.33538188, -0.16535696, -0.45405154,
									 -1.79381571, -1.07692663, -0.70618966, -0.36408311,  0.05569324,
									  0.04228513, -0.42634065, -0.60876426, -0.08098491,  0.08210213,
									  0.15853188,  0.48620148,  1.21346415,  2.06922078,  2.19146684,
									  1.41725225,  0.7444949 , -0.90753406, -1.71688677, -2.23987508,
									 -1.67742056, -1.26793114, -0.03016832, -0.0385695 ,  0.8976368);

	double s = 0.2;    // RBF kernel tuning parameter
	double r = 0.4;    // regularization tuning parameter
	double lr = 0.001; // gradient descent learning rate
	int n = 5000;      // number of gradient descent iterations

	// fit the model
	Vector coef = RBF_fit(X, y, s, r, lr, n);

	// predict on the training inputs
	Vector pred = RBF_predict(X, X, coef, s);

	// print the learned model coefficients 
	Vector_print(pred);

	// free remaining heap memory
	Matrix_delete(X);
	Vector_delete(y);
	Vector_delete(coef);
	Vector_delete(pred);

	return 0;
}