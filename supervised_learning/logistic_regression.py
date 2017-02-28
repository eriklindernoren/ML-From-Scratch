import sys
import os
import math
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import make_diagonal, normalize, train_test_split
from data_operation import accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


# The sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Gradient of the sigmoid func.
def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


class LogisticRegression():
    def __init__(self):
        self.param = None

    def fit(self, X, y, n_iterations=4):
        # Add dummy ones for bias weights
        X = np.insert(X, 0, 1, axis=1)

        n_samples, n_features = np.shape(X)

        # Initial parameters between [-1/sqrt(N), 1/sqrt(N)]
        a = -1 / math.sqrt(n_features)
        b = -a
        self.param = (b - a) * np.random.random((n_features,)) + a

        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = sigmoid(X.dot(self.param))
            # Make a diagonal matrix of the sigmoid gradient column vector
            diag_gradient = make_diagonal(sigmoid_gradient(X.dot(self.param)))
            # Batch opt:
            self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        # Add dummy ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        # Print a final prediction
        dot = X.dot(self.param)
        y_pred = np.round(sigmoid(dot)).astype(int)
        return y_pred


def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)

if __name__ == "__main__":
    main()
