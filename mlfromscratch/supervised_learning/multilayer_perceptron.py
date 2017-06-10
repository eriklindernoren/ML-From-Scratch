from __future__ import print_function
from sklearn import datasets
import sys
import os
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, categorical_to_binary, normalize, binary_to_categorical
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU
from mlfromscratch.utils.optimizers import GradientDescent
from mlfromscratch.unsupervised_learning import PCA

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

class DenseLayer():
    """A fully-connected NN layer. 

    Parameters:
    -----------
    n_inputs: int
        The number of inputs per neuron.
    n_units: int
        The number of neurons in the layer.
    activation_function: class:
        The activation function that will be used for each neuron. 
        Possible choices: Sigmoid, ELU, ReLU, LeakyReLU, SoftPlus, TanH, SELU
    """
    def __init__(self, n_inputs, n_units, activation_function=LeakyReLU):
        self.activation = activation_function()
        self.layer_input = None

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        a, b = -1 / math.sqrt(n_inputs), 1 / math.sqrt(n_inputs)
        self.W  = (b - a) * np.random.random((n_inputs, n_units)) + a
        self.wb = (b - a) * np.random.random((1, n_units)) + a

    def set_optimizer(self, optimizer):
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def update_weights(self, acc_grad, output=False):

        # The accumulated gradient at the layer
        layer_grad = acc_grad * self.activation.gradient(self.layer_input.dot(self.W) + self.wb)

        # Calculate gradient w.r.t layer weights
        grad_w = self.layer_input.T.dot(layer_grad)
        grad_wb = np.ones((1, np.shape(layer_grad)[0])).dot(layer_grad)

        # Update the layer weights
        self.W = self.W_opt.update(w=self.W, grad_wrt_w=grad_w)
        self.wb = self.wb_opt.update(w=self.wb, grad_wrt_w=grad_wb)

        # Return accumulated gradient for next layer
        acc_grad = layer_grad.dot(self.W.T)
        return acc_grad

    def calc_layer_output(self, layer_input):
        self.layer_input = layer_input
        layer_output = self.activation.function(layer_input.dot(self.W) + self.wb)
        return layer_output



class MultilayerPerceptron():
    """Multilayer Perceptron classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    momentum: float
        A momentum term that helps accelerate the optimization by adding a fraction of the previous
        weight update to the current update.
    early_stopping: boolean
        Whether to stop the training when the validation error has increased for a
        certain amounts of training iterations. Combats overfitting.
    plot_errors: boolean
        True or false depending if we wish to plot the training errors after training.
    """
    def __init__(self, n_iterations=3000, learning_rate=0.0001, momentum=0, early_stopping=False, plot_errors=False):
        self.n_iterations = n_iterations
        self.plot_errors = plot_errors
        self.early_stopping = early_stopping
        self.optimizer = GradientDescent(learning_rate, momentum)
        self.layers = []

    def add(self, layer):
        layer.set_optimizer(self.optimizer)
        self.layers.append(layer)

    def fit(self, X, y):
        X_train = X
        y_train = y

        if self.early_stopping:
            # Split the data into training and validation sets
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1)
            y_validate = categorical_to_binary(y_validate)

        # Convert the nominal y values to binary
        y_train = categorical_to_binary(y_train)

        # Error history
        training_errors = []
        validation_errors = []
        iter_with_rising_val_error = 0

        bar = progressbar.ProgressBar(widgets=bar_widgets)

        for i in bar(range(self.n_iterations)):

            # Calculate output
            y_pred = self._forward_pass(X_train)

            # Determine the error
            mse = np.mean(np.power(y_train - y_pred, 2))
            training_errors.append(mse)

            # Update the NN weights
            self._backward_pass(loss_grad=-2*(y_train - y_pred))


            if self.early_stopping:
                # Calculate the validation error
                e = y_validate - self._forward_pass(X_validate)
                mse = np.mean(np.power(e, 2))
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

    def _forward_pass(self, X):
        # Calculate the output of the NN. The output of layer l1 becomes the
        # input of the following layer l2
        layer_output = X
        for layer in self.layers:
            layer_output = layer.calc_layer_output(layer_output)

        return layer_output

    def _backward_pass(self, loss_grad):
        # Propogate the gradient 'backwards' and update the
        # weights by moving against the gradient of the loss func.
        acc_grad = loss_grad
        for layer in reversed(self.layers):
            acc_grad = layer.update_weights(acc_grad)


    # Use the trained model to predict labels of X
    def predict(self, X):
        output = self._forward_pass(X)
        # Predict as the indices of the highest valued outputs
        y_pred = np.argmax(output, axis=1)
        return y_pred


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    n_samples, n_features = np.shape(X)
    n_hidden, n_output = 128, 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # MLP
    clf = MultilayerPerceptron(n_iterations=10000,
                            learning_rate=0.0001, 
                            momentum=0.4,
                            early_stopping=True,
                            plot_errors=True)

    clf.add(DenseLayer(n_inputs=n_features, n_units=n_hidden))
    clf.add(DenseLayer(n_inputs=n_hidden, n_units=n_hidden))
    clf.add(DenseLayer(n_inputs=n_hidden, n_units=n_output))   
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=np.unique(y))

if __name__ == "__main__":
    main()
