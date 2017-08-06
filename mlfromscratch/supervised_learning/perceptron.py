from __future__ import print_function
import sys
import os
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, categorical_to_binary, normalize, binary_to_categorical
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU
from mlfromscratch.utils.optimizers import GradientDescent
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils import Plot


class Perceptron():
    """The Perceptron. One layer neural network classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class:
        The activation that shall be used for each neuron. 
        Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, learning_rate=0.01):
        self.W = None           # Output layer weights
        self.w0 = None          # Bias weights
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.activation = activation_function()

    def fit(self, X, y):
        X_train = X
        y_train = y

        # One-hot encoding of nominal y-values
        y_train = categorical_to_binary(y_train)

        n_samples, n_features = np.shape(X_train)
        n_outputs = np.shape(y_train)[1]

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = -1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in range(self.n_iterations):
            # Calculate outputs
            linear_output = np.dot(X_train, self.W) + self.w0
            y_pred = self.activation.function(linear_output)

            # Calculate the loss gradient
            error_gradient = -2 * (y_train - y_pred) * \
                self.activation.gradient(linear_output)

            # Calculate the gradient of the loss with respect to each weight term
            grad_wrt_w = X_train.T.dot(error_gradient)
            grad_wrt_w0 = np.ones((1, n_samples)).dot(error_gradient)

            # Update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate  * grad_wrt_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        output = self.activation.function(np.dot(X, self.W) + self.w0)
        # Predict as the indices of the largest outputs
        y_pred = np.argmax(output, axis=1)
        return y_pred


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=5000,
        learning_rate=0.001, 
        activation_function=Sigmoid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Perceptron", accuracy=accuracy, legend_labels=np.unique(y))


if __name__ == "__main__":
    main()