from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sys
import os
import math
# Import helper functions
from mlfromscratch.utils.data_manipulation import k_fold_cross_validation_sets
from mlfromscratch.utils.data_manipulation import train_test_split
from mlfromscratch.utils.data_operation import mean_squared_error
from mlfromscratch.utils.loss_functions import SquareLoss


class RidgeRegression():
    """Linear regression model with a regularization factor.
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
    def __init__(self, reg_factor, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.w = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.reg_factor = reg_factor
        self.square_loss = SquareLoss()

    def fit(self, X, y):
        # Insert dummy ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        n_features = np.shape(X)[1]

        # Get weights by gradient descent opt.
        if self.gradient_descent:
            # Initial weights randomly [0, 1]
            self.w = np.random.random((n_features, ))
            # Do gradient descent for n_iterations
            for _ in range(self.n_iterations):
                grad_w = self.square_loss.gradient(y, X, self.w) + self.reg_factor * self.w
                self.w -= self.learning_rate * grad_w
        # Get weights by least squares (by pseudoinverse)
        else:
            U, S, V = np.linalg.svd(
                X.T.dot(X) + self.reg_factor * np.identity(n_features))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred



def main():
    # Load the diabetes dataset
    X, y = datasets.make_regression(n_features=1, n_samples=100, bias=3, noise=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Finding regularization constant using cross validation
    lowest_error = float("inf")
    best_reg_factor = None
    print ("Finding regularization constant using cross validation:")
    k = 10
    for reg_factor in np.arange(0, 0.3, 0.001):
        cross_validation_sets = k_fold_cross_validation_sets(
            X_train, y_train, k=k)
        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            clf = RidgeRegression(reg_factor=reg_factor)
            clf.fit(_X_train, _y_train)
            y_pred = clf.predict(_X_test)
            _mse = mean_squared_error(_y_test, y_pred)
            mse += _mse
        mse /= k

        # Print the mean squared error
        print ("\tMean Squared Error: %s (regularization: %s)" % (mse, reg_factor))

        # Save reg. constant that gave lowest error
        if mse < lowest_error:
            best_reg_factor = reg_factor
            lowest_error = mse

    # Make final prediction
    clf = RidgeRegression(reg_factor=best_reg_factor)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s (given by reg. factor: %s)" % (lowest_error, best_reg_factor))
    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.plot(X_test[:, 0], y_pred, color='blue', linewidth=3)
    plt.title("Ridge Regression (%.2f MSE)" % mse)
    plt.show()

if __name__ == "__main__":
    main()
