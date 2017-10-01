from __future__ import print_function, division
import numpy as np
import math
from mlfromscratch.utils import normalize, polynomial_features

class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
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
    def __init__(self, n_iterations, learning_rate, gradient_descent):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones as first column (for bias weights)
        X = np.insert(X, 0, 1, axis=1)
        # Get weights by gradient descent opt.
        if self.gradient_descent:
            self.training_errors = []
            self.initialize_weights(n_features=X.shape[1])
            # Do gradient descent for n_iterations
            for _ in range(self.n_iterations):
                y_pred = X.dot(self.w)
                # Calculate l2 loss
                mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
                self.training_errors.append(mse)
                # Gradient of l2 loss w.r.t w
                grad_w = - (y - y_pred).dot(X) + self.regularization.grad(self.w)
                # Update the weights
                self.w -= self.learning_rate * grad_w
        # Get weights by least squares (using Moore-Penrose pseudoinverse)
        else:
            n_features = np.shape(X)[1]
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
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate, 
                                            gradient_descent=gradient_descent)

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
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                learning_rate=learning_rate, 
                                                gradient_descent=gradient_descent)

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
        # Ridge Regression
        self.regularization = l2_regularization(reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, 
                                            learning_rate, 
                                            gradient_descent)

class LassoRegression(Regression):
    """Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.
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
        # Lasso Regression
        self.regularization = l1_regularization(reg_factor)
        super(LassoRegression, self).__init__(n_iterations, 
                                            learning_rate, 
                                            gradient_descent)

    def fit(self, X, y):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X_transformed, y)

    def predict(self, X):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X_transformed)

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
        # Ridge Regression
        self.regularization = l2_regularization(reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations, 
                                                        learning_rate, 
                                                        gradient_descent)

    def fit(self, X, y):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X_transformed, y)

    def predict(self, X):
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X_transformed)


