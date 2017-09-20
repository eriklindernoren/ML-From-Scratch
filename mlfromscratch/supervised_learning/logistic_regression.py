from __future__ import print_function
import numpy as np
import math
from mlfromscratch.utils import make_diagonal, Plot
from mlfromscratch.deep_learning.activation_functions import Sigmoid


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

    def fit(self, X, y, n_iterations=4000):
        # Add dummy ones for bias weights
        X = np.insert(X, 0, 1, axis=1)

        n_samples, n_features = np.shape(X)

        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid.function(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with 
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
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
        
