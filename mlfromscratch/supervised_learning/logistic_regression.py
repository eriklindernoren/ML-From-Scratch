from __future__ import print_function
import sys
import os
import math
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.utils.data_manipulation import make_diagonal, normalize, train_test_split
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.activation_functions import Sigmoid
from mlfromscratch.utils.loss_functions import LogisticLoss
from mlfromscratch.utils.optimizers import GradientDescent
from mlfromscratch.unsupervised_learning import PCA


class LogisticRegression():
    """The Logistic Regression classifier. 
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()
        self.log_loss = LogisticLoss()


    def fit(self, X, y, n_iterations=4000):
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
            y_pred = self.sigmoid.function(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with 
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * self.log_loss.gradient(y, X, self.param)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        # Add dummy ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        # Print a final prediction
        dot = X.dot(self.param)
        y_pred = np.round(self.sigmoid.function(dot)).astype(int)
        return y_pred
        

def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)

if __name__ == "__main__":
    main()
