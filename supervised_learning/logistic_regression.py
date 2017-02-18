import sys, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import accuracy_score, make_diagonal, normalize
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

# The sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))

class LogisticRegression():
    def __init__(self):
        self.param = None

    def fit(self, X, y, n_iterations=4):
        X_train = np.array(X, dtype=float)
        # Add one to take bias weights into consideration
        X_train = np.insert(X_train, 0, 1, axis=1)
        y_train = np.atleast_1d(y)

        n_features = len(X_train[0])

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
        a = -1/math.sqrt(n_features)
        b = -a
        self.param = (b-a)*np.random.random((len(X_train[0]),)) + a

        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            dot = X_train.dot(self.param)
            y_pred = sigmoid(dot)

            # Make a diagonal matrix of the sigmoid gradient column vector
            diag_gradient = make_diagonal(sigmoid_gradient(dot))

            # Batch opt:
            # (X^T * diag(sigm*(1 - sigm) * X) * X^T * (diag(sigm*(1 - sigm) * X * param + Y - Y_pred)
            self.param = np.linalg.inv(X_train.T.dot(diag_gradient).dot(X_train)).dot(X_train.T).dot(diag_gradient.dot(X_train).dot(self.param) + y_train - y_pred)

    def predict(self, X):
        X_test = np.array(X, dtype=float)
        # Add ones to take bias weights into consideration
        X_test = np.insert(X_test, 0, 1, axis=1)
        # Print a final prediction
        dot = X_test.dot(self.param)
        y_pred = np.round(sigmoid(dot)).astype(int)
        return y_pred

# Demo
def main():
    df = pd.read_csv(dir_path + "/../data/diabetes.csv")

    Y = df["Outcome"]
    y_train = Y[:-300].as_matrix()
    y_test = Y[-300:].as_matrix()

    X = df.drop("Outcome", axis=1)
    X_train = np.insert(normalize(X[:-300].as_matrix()),0,1,axis=1)
    X_test = np.insert(normalize(X[-300:].as_matrix()),0,1,axis=1)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

     # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)

if __name__ == "__main__": main()

