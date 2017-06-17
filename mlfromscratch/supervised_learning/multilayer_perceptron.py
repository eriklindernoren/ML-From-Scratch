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
from mlfromscratch.utils.data_manipulation import get_random_subsets, shuffle_data, normalize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU, Softmax
from mlfromscratch.utils.optimizers import GradientDescent, GradientDescent_, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.utils.loss_functions import CrossEntropy
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.utils import Plot


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
        Possible choices: Sigmoid, ELU, ReLU, LeakyReLU, SoftPlus, TanH, SELU, Softmax
    """
    def __init__(self, n_inputs, n_units, activation_function=SELU):
        self.activation = activation_function()
        self.layer_input = None
        self.initialized = False
        self.n_inputs, self.n_units = n_inputs, n_units
        self.W = None
        self.wb = None

    def initialize(self, optimizer):
        # Initialize the weights
        a, b = -1 / math.sqrt(self.n_inputs), 1 / math.sqrt(self.n_inputs)
        self.W  = (b - a) * np.random.random((self.n_inputs, self.n_units)) + a
        self.wb = (b - a) * np.random.random((1, self.n_units)) + a
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def backward_pass(self, acc_grad, output=False):

        # The accumulated gradient at the layer
        layer_grad = lambda w, b: acc_grad * self.activation.gradient(self.layer_input.dot(w) + b)

        # Calculate gradient w.r.t layer weights
        grad_w = lambda w: self.layer_input.T.dot(layer_grad(w, self.wb))
        grad_wb = lambda b: np.ones((1, np.shape(layer_grad(self.W, b))[0])).dot(layer_grad(self.W, b))

        # Update the layer weights
        self.W = self.W_opt.update(w=self.W, grad_func=grad_w)
        self.wb = self.wb_opt.update(w=self.wb, grad_func=grad_wb)

        # Return accumulated gradient for next layer
        acc_grad = layer_grad(self.W, self.wb).dot(self.W.T)
        return acc_grad

    def forward_pass(self, layer_input):
        self.layer_input = layer_input
        layer_output = self.activation.function(layer_input.dot(self.W) + self.wb)
        return layer_output



class MultilayerPerceptron():
    """Multilayer Perceptron classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    batch_size: int
        The size of the batches that the model will train on at a time.
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    validation: tuple
        A tuple containing validation data and labels
    """
    def __init__(self, n_iterations, batch_size, optimizer, validation_data=None):
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.cross_ent = CrossEntropy()
        self.batch_size = batch_size
        self.X_val = self.y_val = np.empty([])
        if validation_data:
            self.X_val, self.y_val = validation_data
            self.y_val = categorical_to_binary(self.y_val.astype("int"))

    def add(self, layer):
        layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def fit(self, X, y):
        # Convert the categorical data to binary
        y = categorical_to_binary(y.astype("int"))

        n_samples, n_features = np.shape(X)
        n_batches = int(n_samples / self.batch_size)

        bar = progressbar.ProgressBar(widgets=bar_widgets)
        for i in bar(range(self.n_iterations)):
            X_, y_ = shuffle_data(X, y)

            batch_t_error = 0   # Mean batch training error
            for idx in np.array_split(np.arange(n_samples), n_batches):
                X_batch, y_batch = X_[idx], y_[idx]

                # Calculate output
                y_pred = self._forward_pass(X_batch)

                # Calculate the cross entropy training loss
                loss = np.mean(self.cross_ent.loss(y_batch, y_pred))
                batch_t_error += loss

                loss_grad = self.cross_ent.gradient(y_batch, y_pred)

                # Update the NN weights
                self._backward_pass(loss_grad=loss_grad)

            batch_t_error /= n_batches
            self.errors["training"].append(batch_t_error)
            if self.X_val.any():
                # Calculate the validation error
                y_val_p = self._forward_pass(self.X_val)
                loss = np.mean(self.cross_ent.loss(self.y_val, y_val_p))
                self.errors["validation"].append(loss)

    def _forward_pass(self, X):
        # Calculate the output of the NN. The output of layer l1 becomes the
        # input of the following layer l2
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output)

        return layer_output

    def _backward_pass(self, loss_grad):
        # Propogate the gradient 'backwards' and update the weights
        # in each layer
        acc_grad = loss_grad
        for layer in reversed(self.layers):
            acc_grad = layer.backward_pass(acc_grad)

    def plot_errors(self):
        if self.errors["training"]:
            n = len(self.errors["training"])
            if self.errors["validation"]:
                # Training and validation error plot
                training, = plt.plot(range(n), self.errors["training"], label="Training Error")
                validation, = plt.plot(range(n), self.errors["validation"], label="Validation Error")
                plt.legend(handles=[training, validation])
            else:
                training, = plt.plot(range(n), self.errors["training"], label="Training Error")
                plt.legend(handles=[training])
            plt.title("Error Plot")
            plt.ylabel('Error')
            plt.xlabel('Iterations')
            plt.show()

    # Use the trained model to predict labels of X
    def predict(self, X):
        output = self._forward_pass(X)
        # Return the sample with the highest output
        return np.argmax(output, axis=1)


def main():

    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    n_samples, n_features = np.shape(X)
    n_hidden, n_output = 128, 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    optimizer = GradientDescent_(learning_rate=0.005, momentum=0.9)

    # MLP
    clf = MultilayerPerceptron(n_iterations=150,
                            batch_size=64,
                            optimizer=optimizer,
                            validation_data=(X_test, y_test))

    clf.add(DenseLayer(n_inputs=n_features, n_units=n_hidden))
    # clf.add(DenseLayer(n_inputs=n_hidden, n_units=n_hidden))
    clf.add(DenseLayer(n_inputs=n_hidden, n_units=n_output, activation_function=Softmax))  
    
    clf.fit(X_train, y_train)
    clf.plot_errors()

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=np.unique(y))

if __name__ == "__main__":
    main()
