from __future__ import print_function
import numpy as np
import math
from mlfromscratch.utils import normalize, polynomial_features


class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, reg_factor, n_iterations, learning_rate, gradient_descent):
        self.w = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.reg_factor = reg_factor

    def fit(self, X, y):
        # Insert constant ones as first column (for bias weights)
        X = np.insert(X, 0, 1, axis=1)
        n_features = np.shape(X)[1]
        # Get weights by gradient descent opt.
        if self.gradient_descent:
            # Initial weights randomly [-1/N, 1/N]
            limit = 1 / math.sqrt(n_features)
            self.w = np.random.uniform(-limit, limit, (n_features, ))
            # Do gradient descent for n_iterations
            for _ in range(self.n_iterations):
                y_pred = X.dot(self.w)
                # Gradient of l2 loss w.r.t w
                grad_w = - (y - y_pred).dot(X) + self.reg_factor * self.w
                # Update the weights
                self.w -= self.learning_rate * grad_w
        # Get weights by least squares (using Moore-Penrose pseudoinverse)
        else:
            U, S, V = np.linalg.svd(X.T.dot(X) + self.reg_factor * np.identity(n_features))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations=1000, learning_rate=0.001, gradient_descent=True):
        super(LinearRegression, self).__init__(reg_factor=0, n_iterations=n_iterations, \
                                learning_rate=learning_rate, gradient_descent=gradient_descent)

class PolynomialRegression(Regression):
    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The power of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001, gradient_descent=True):
        self.degree = degree
        super(PolynomialRegression, self).__init__(reg_factor=0, n_iterations=n_iterations, \
                                learning_rate=learning_rate, gradient_descent=gradient_descent)

    def fit(self, X, y):
        X_transformed = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X_transformed, y)

    def predict(self, X):
        X_transformed = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X_transformed)

class RidgeRegression(Regression):
    """Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001, gradient_descent=True):
        super(RidgeRegression, self).__init__(reg_factor, n_iterations, learning_rate, gradient_descent)

class PolynomialRidgeRegression(Regression):
    """Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression.
    Parameters:
    -----------
    degree: int
        The power of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        self.degree = degree
        super(PolynomialRidgeRegression, self).__init__(reg_factor, n_iterations, learning_rate, gradient_descent)

    def fit(self, X, y):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X_transformed, y)

    def predict(self, X):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X_transformed)

