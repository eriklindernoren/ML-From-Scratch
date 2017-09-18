from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.data_operation import mean_squared_error, calculate_variance
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import RegressionTree

def main():

    print ("-- Regression Tree --")

    X, y = datasets.make_regression(n_features=1, n_samples=100, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.3)

    clf = RegressionTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.title("Regression Tree (%.2f MSE)" % mse)
    plt.show()


if __name__ == "__main__":
    main()