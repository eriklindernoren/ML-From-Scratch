from __future__ import print_function
from sklearn import datasets
import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, to_categorical, normalize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils import Plot
from mlfromscratch.utils.activation_functions import Sigmoid, Softmax
from mlfromscratch.utils.loss_functions import CrossEntropy

class MultilayerPerceptron():
    """Multilayer Perceptron classifier. A neural network with one hidden layer.
    Unrolled to display the whole forward pass and backward pass.

    Parameters:
    -----------
    n_hidden: int:
        The number of processing nodes (neurons) in the hidden layer. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    loss: class
        The loss function used. Either CrossEntropy or SquareLoss.
    """
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        self.W = None   # Hidden layer weights
        self.w0 = None  # Hidden layer bias
        self.V = None   # Output layer weights
        self.v0 = None  # Output layer bias

    def initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit   = 1 / math.sqrt(n_features)
        self.W  = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        limit   = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, X, y):

        self.initialize_weights(X, y)

        for i in range(self.n_iterations):

            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation.function(hidden_input)
            # OUTPUT LAYER
            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation.function(output_layer_input)

            # ...............
            #  Backward Pass
            # ...............

            # OUTPUT LAYER
            # Grad. w.r.t input of output layer
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            # HIDDEN LAYER
            # Grad. w.r.t input of hidden layer
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            self.V  -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W  -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        # Forward pass:
        # Calculate hidden layer
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation.function(hidden_input)
        # Calculate output layer
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation.function(output_layer_input)
        return y_pred


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # Convert the nominal y values to binary
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # MLP
    clf = MultilayerPerceptron(n_hidden=12,
        n_iterations=1000,
        learning_rate=0.01)

    clf.fit(X_train, y_train)
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=np.unique(y))

if __name__ == "__main__":
    main()