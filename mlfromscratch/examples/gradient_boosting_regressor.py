from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import line_search
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize, to_categorical
from mlfromscratch.utils.data_operation import mean_squared_error, accuracy_score
from mlfromscratch.utils.loss_functions import SquareLoss
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.supervised_learning import GradientBoostingRegressor
from mlfromscratch.utils import Plot


def main():
    print ("-- Gradient Boosting Regression --")

    X, y = datasets.make_regression(n_features=1, n_samples=150, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.5)

    clf = GradientBoostingRegressor(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.title("Gradient Boosting Regression (%.2f MSE)" % mse)
    plt.show()


if __name__ == "__main__":
    main()