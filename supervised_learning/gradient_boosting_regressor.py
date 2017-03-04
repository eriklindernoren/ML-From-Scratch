from __future__ import division
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, standardize
from data_operation import mean_squared_error
from regression_tree import RegressionTree


class GradientBoostingRegressor():
    def __init__(self, n_estimators=20, learning_rate=0.5, min_samples_split=10,
                 min_var_red=1e-4, max_depth=4):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_var_red = min_var_red              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.init_estimate = None

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_var_red=min_var_red,
                    max_depth=self.max_depth))

    def fit(self, X, y):
        # Set initial predictions to zero
        self.init_estimate = np.mean(y)
        y_pred = self.init_estimate * np.ones(np.shape(y))
        for tree in self.trees:
            # Calculate the gradient of the loss (MSE)
            residuals = -(y - y_pred)
            tree.fit(X, residuals)
            residual_pred = tree.predict(X)
            # Update y prediction by the estimated residual value
            y_pred -= np.multiply(self.learning_rate, residual_pred)

    def predict(self, X):
        y_pred = self.init_estimate * np.ones(np.shape(X)[0])
        for tree in self.trees:
            y_pred -= np.multiply(self.learning_rate, tree.predict(X))
        return y_pred



def main():

    X, y = datasets.make_regression(n_features=1, n_samples=200, bias=100, noise=1)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.4)

    clf = GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print the mean squared error
    print "Mean Squared Error:", mean_squared_error(y_test, y_pred)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.show()


if __name__ == "__main__":
    main()