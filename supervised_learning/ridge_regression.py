import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sys
import os
import math
# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import k_fold_cross_validation_sets
from data_manipulation import train_test_split
from data_operation import mean_squared_error


class RidgeRegression():
    def __init__(self, delta, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.w = None                               # Weights
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent    # Opt. method. If False => Least squares
        self.delta = delta                          # Regularization constant

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
                w_gradient = X.T.dot(X.dot(self.w) - y) + self.delta * self.w
                self.w -= self.learning_rate * w_gradient
        # Get weights by least squares (by pseudoinverse)
        else:
            U, S, V = np.linalg.svd(
                X.T.dot(X) + self.delta * np.identity(n_features))
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
    print "Finding regularization constant using cross validation:"
    k = 10
    for regularization_factor in np.arange(0, 0.3, 0.001):
        cross_validation_sets = k_fold_cross_validation_sets(
            X_train, y_train, k=k)
        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            clf = RidgeRegression(delta=regularization_factor)
            clf.fit(_X_train, _y_train)
            y_pred = clf.predict(_X_test)
            _mse = mean_squared_error(_y_test, y_pred)
            mse += _mse
        mse /= k

        # Print the mean squared error
        print "\tMean Squared Error: %s (regularization: %s)" % (mse, regularization_factor)

        # Save reg. constant that gave lowest error
        if mse < lowest_error:
            best_reg_factor = regularization_factor
            lowest_error = mse

    # Make final prediction
    clf = RidgeRegression(delta=best_reg_factor)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print "Mean squared error: %s (given by reg. factor: %s)" % (lowest_error, best_reg_factor)
    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.plot(X_test[:, 0], y_pred, color='blue', linewidth=3)
    plt.show()

if __name__ == "__main__":
    main()
