from __future__ import print_function
import sys
import os
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, categorical_to_binary, normalize, binary_to_categorical
from data_operation import accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Gradient of activation function
def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron():
    def __init__(self, n_iterations=20000,
            learning_rate=0.01, plot_errors=False):
        self.W = None           # Output layer weights
        self.biasW = None       # Bias weights
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.plot_errors = plot_errors

    def fit(self, X, y):
        y = categorical_to_binary(y)

        n_samples, n_features = np.shape(X)
        n_outputs = np.shape(y)[1]

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)]
        a = -1 / math.sqrt(n_features)
        b = -a
        self.W = (b - a) * np.random.random((n_features, n_outputs)) + a
        self.biasW = (b - a) * np.random.random((1, n_outputs)) + a

        errors = []
        for i in range(self.n_iterations):
            # Calculate outputs
            neuron_input = np.dot(X, self.W) + self.biasW
            neuron_output = sigmoid(neuron_input)

            # Training error
            error = y - neuron_output
            mse = np.mean(np.power(error, 2))
            errors.append(mse)

            # Calculate the loss gradient
            w_gradient = -2 * (y - neuron_output) * \
                sigmoid_gradient(neuron_input)
            bias_gradient = w_gradient

            # Update weights
            self.W -= self.learning_rate * X.T.dot(w_gradient)
            self.biasW -= self.learning_rate * \
                np.ones((1, n_samples)).dot(bias_gradient)

        # Plot the training error
        if self.plot_errors:
            plt.plot(range(self.n_iterations), errors)
            plt.ylabel('Training Error')
            plt.xlabel('Iterations')
            plt.title("Training Error Plot")
            plt.show()

    # Use the trained model to predict labels of X
    def predict(self, X):
        # Set the class labels to the highest valued outputs
        y_pred = np.argmax(sigmoid(np.dot(X, self.W) + self.biasW), axis=1)
        return y_pred


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=4000,
        learning_rate=0.01, 
        plot_errors=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, title="Perceptron", accuracy=accuracy)


if __name__ == "__main__":
    main()
