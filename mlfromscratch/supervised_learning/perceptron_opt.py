from __future__ import print_function
import sys
import os
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, categorical_to_binary, normalize, binary_to_categorical
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU
from mlfromscratch.utils.optimizers import GradientDescent, GradientDescent_, NesterovAcceleratedGradient, Adagrad, Adadelta, RMSprop, Adam
from mlfromscratch.unsupervised_learning import PCA


class Perceptron():
    """The Perceptron. One layer neural network classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class:
        The activation that shall be used for each neuron. 
        Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH
    optimizer: class:
        The optimization method that will be used to find the weights that minimizes the
        loss.
        Possible choices: GradientDescent, NesterovAcceleratedGradient, Adagrad, Adadelta, RMSprop, Adam
    early_stopping: boolean
        Whether to stop the training when the validation error has increased for a
        certain amounts of training iterations. Combats overfitting.
    plot_errors: boolean
        True or false depending if we wish to plot the training errors after training.
    """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, optimizer=GradientDescent(0.01, 0.3),
        early_stopping=False, plot_errors=False):
        self.W = None           # Output layer weights
        self.w0 = None          # Bias weights
        self.n_iterations = n_iterations
        self.plot_errors = plot_errors
        self.early_stopping = early_stopping
        # Activation function of each neuron
        self.activation = activation_function()
        # Optimization methods for each parameter set
        self.w_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def fit(self, X, y):
        X_train = X
        y_train = y

        if self.early_stopping:
            # Split the data into training and validation sets
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1)
            y_validate = categorical_to_binary(y_validate)

        # Convert the nominal y values to binary
        y_train = categorical_to_binary(y_train)

        n_samples, n_features = np.shape(X_train)
        n_outputs = np.shape(y_train)[1]

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)]
        a = -1 / math.sqrt(n_features)
        b = -a
        self.W = (b - a) * np.random.random((n_features, n_outputs)) + a
        self.w0 = (b - a) * np.random.random((1, n_outputs)) + a

        # Error history
        training_errors = []
        validation_errors = []
        iter_with_rising_val_error = 0

        # Lambda function that calculates the neuron outputs
        neuron_output = lambda w, b: self.activation.function(np.dot(X_train, w) + b)

        # Lambda function that calculates the loss gradient
        loss_grad = lambda w, b: -2 * (y_train - neuron_output(w, b)) * \
            self.activation.gradient(np.dot(X_train, w) + b)

        # Lambda functions that calculates the gradient of the loss with 
        # respect to each weight term. Allows for computation of loss gradient at different
        # coordinates.
        grad_func_wrt_w = lambda w: X_train.T.dot(loss_grad(w, self.w0))
        grad_func_wrt_w0 = lambda b: np.ones((1, n_samples)).dot(loss_grad(self.W, b))

        # Optimize paramaters for n_iterations
        for i in range(self.n_iterations):

            # Training error
            error = y_train - neuron_output(self.W, self.w0)
            mse = np.mean(np.power(error, 2))
            training_errors.append(mse)

            # Update weights
            self.W = self.w_opt.update(w=self.W, grad_func=grad_func_wrt_w)
            self.w0 = self.w0_opt.update(w=self.w0, grad_func=grad_func_wrt_w0)

            if self.early_stopping:
                # Calculate the validation error
                error = y_validate - self._calculate_output(X_validate)
                mse = np.mean(np.power(error, 2))
                validation_errors.append(mse)

                # If the validation error is larger than the previous iteration increase
                # the counter
                if len(validation_errors) > 1 and validation_errors[-1] > validation_errors[-2]:
                    iter_with_rising_val_error += 1
                    # If the validation error has been for more than 50 iterations
                    # stop training to avoid overfitting
                    if iter_with_rising_val_error > 50:
                        break
                else:
                    iter_with_rising_val_error = 0

        # Plot the training error
        if self.plot_errors:
            if self.early_stopping:
                # Training and validation error plot
                training, = plt.plot(range(i+1), training_errors, label="Training Error")
                validation, = plt.plot(range(i+1), validation_errors, label="Validation Error")
                plt.legend(handles=[training, validation])
            else:
                training, = plt.plot(range(i+1), training_errors, label="Training Error")
                plt.legend(handles=[training])
            plt.title("Error Plot")
            plt.ylabel('Error')
            plt.xlabel('Iterations')
            plt.show()

    def _calculate_output(self, X):
        # Calculate the output layer values
        output = self.activation.function(np.dot(X, self.W) + self.w0)

        return output

    # Use the trained model to predict labels of X
    def predict(self, X):
        output = self._calculate_output(X)
        # Predict as the indices of the largest outputs
        y_pred = np.argmax(output, axis=1)
        return y_pred


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Optimization method for finding weights that minimizes loss
    optimizer = RMSprop(learning_rate=0.001)

    # Perceptron
    clf = Perceptron(n_iterations=5000,
        activation_function=SELU,
        optimizer=optimizer,
        early_stopping=True,
        plot_errors=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, title="Perceptron", accuracy=accuracy, legend_labels=np.unique(y))


if __name__ == "__main__":
    main()
