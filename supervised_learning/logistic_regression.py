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
        x_train = normalize(np.array(X, dtype=float))
        x_train = np.insert(x_train, 0, 1, axis=1)
        y_train = np.atleast_1d(y)

        n_features = len(x_train[0])

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
        a = -1/math.sqrt(n_features)
        b = -a
        self.param = (b-a)*np.random.random((len(x_train[0]),)) + a

        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            dot = x_train.dot(self.param)
            y_pred = sigmoid(dot)

            # Make a diagonal matrix of the sigmoid gradient column vector
            diag_gradient = make_diagonal(sigmoid_gradient(dot))

            # Batch opt:
            # (X^T * diag(sigm*(1 - sigm) * X) * X^T * (diag(sigm*(1 - sigm) * X * param + Y - Y_pred)
            self.param = np.linalg.inv(x_train.T.dot(diag_gradient).dot(x_train)).dot(x_train.T).dot(diag_gradient.dot(x_train).dot(self.param) + y_train - y_pred)

    def predict(self, X):
        x_test = normalize(np.array(X, dtype=float))
        x_test = np.insert(x_test, 0, 1, axis=1)
        # Print a final prediction
        dot = x_test.dot(self.param)
        y_pred = np.round(sigmoid(dot)).astype(int)
        return y_pred

# Demo of Logistic Regression
def main():
    df = pd.read_csv(dir_path + "/../data/diabetes.csv")

    Y = df["Outcome"]
    y_train = Y[:-300].as_matrix()
    y_test = Y[-300:].as_matrix()

    X = df.drop("Outcome", axis=1)
    x_train = np.insert(normalize(X[:-300].as_matrix()),0,1,axis=1)
    x_test = np.insert(normalize(X[-300:].as_matrix()),0,1,axis=1)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

     # Reduce dimension to two using PCA and plot the results
    pca = PCA(n_components=2)
    X_transformed = pca.transform(x_test)
    x1 = X_transformed[:,0]
    x2 = X_transformed[:,1]

    plt.scatter(x1,x2,c=y_pred)
    plt.show()

if __name__ == "__main__": main()

