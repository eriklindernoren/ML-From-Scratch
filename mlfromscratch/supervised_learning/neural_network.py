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
from mlfromscratch.utils.data_manipulation import train_test_split, to_categorical, normalize
from mlfromscratch.utils.data_manipulation import get_random_subsets, shuffle_data, normalize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.optimizers import GradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.utils.loss_functions import CrossEntropy, SquareLoss
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.utils import Plot
from mlfromscratch.utils.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, ZeroPadding2D



class NeuralNetwork():
    """Neural Network.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the models performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels
    """
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.X_val = np.empty([])
        self.y_val = np.empty([])
        if validation_data:
            self.X_val, self.y_val = validation_data
            self.y_val = to_categorical(self.y_val.astype("int"))

    # Method which enables freezing of the weights of the network's layers.
    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        # If the first layer has been added set the input shape
        # as the output shape of the previous layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        # Calculate output
        y_pred = self._forward_pass(X)
        # Calculate the training loss
        loss = np.mean(self.loss_function.loss(y, y_pred))
        # Calculate the accuracy of the prediction
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backprop. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc


    def fit(self, X, y, n_iterations, batch_size):

        # Convert to one-hot encoding
        y = to_categorical(y.astype("int"))

        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)

        bar = progressbar.ProgressBar(widgets=bar_widgets)
        for _ in bar(range(n_iterations)):
            idx = range(n_samples)
            np.random.shuffle(idx)

            batch_t_error = 0   # Mean batch training error
            for i in range(n_batches):
                X_batch = X[idx[i*batch_size:(i+1)*batch_size]]
                y_batch = y[idx[i*batch_size:(i+1)*batch_size]]
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_t_error += loss

            # Save the epoch mean error
            self.errors["training"].append(batch_t_error / n_batches)
            if self.X_val.any():
                # Calculate the validation error
                y_val_p = self._forward_pass(self.X_val)
                validation_loss = np.mean(self.loss_function.loss(self.y_val, y_val_p))
                self.errors["validation"].append(validation_loss)

    def _forward_pass(self, X, training=True):
        # Calculate the output of the NN. The output of layer l1 becomes the
        # input of the following layer l2
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

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
        return self._forward_pass(X, training=False)


def main():

    data = datasets.load_digits()
    X = data.data
    y = data.target

    n_samples = np.shape(X)
    n_hidden = 512

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    optimizer = Adam()

    #-----
    # MLP
    #-----

    # clf = NeuralNetwork(optimizer=optimizer,
    #                     loss=CrossEntropy,
    #                     validation_data=(X_test, y_test))

    # clf.add(Dense(n_hidden, input_shape=(8*8,)))
    # clf.add(Activation('leaky_relu'))
    # clf.add(Dense(n_hidden))
    # clf.add(Activation('leaky_relu'))
    # clf.add(Dropout(0.25))
    # clf.add(Dense(n_hidden))
    # clf.add(Activation('leaky_relu'))
    # clf.add(Dropout(0.25))
    # clf.add(Dense(n_hidden))
    # clf.add(Activation('leaky_relu'))
    # clf.add(Dropout(0.25))
    # clf.add(Dense(10))
    # clf.add(Activation('softmax'))
    
    # clf.fit(X_train, y_train, n_iterations=50, batch_size=256)
    # clf.plot_errors()

    # y_pred = np.argmax(clf.predict(X_test), axis=1)

    # accuracy = accuracy_score(y_test, y_pred)
    # print ("Accuracy:", accuracy)

    #----------
    # Conv Net
    #----------

    X_train = X_train.reshape((-1,1,8,8))
    X_test = X_test.reshape((-1,1,8,8))

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), padding=1, input_shape=(1,8,8)))
    clf.add(Activation('relu'))
    clf.add(Conv2D(n_filters=32, filter_shape=(3,3), padding=1))
    clf.add(Activation('relu'))
    clf.add(Flatten())
    clf.add(Dense(128))
    clf.add(Activation('relu'))
    clf.add(Dense(10))
    clf.add(Activation('softmax'))
    
    clf.fit(X_train, y_train, n_iterations=50, batch_size=256)
    clf.plot_errors()

    y_pred = np.argmax(clf.predict(X_test), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    X_test = X_test.reshape(-1, 8*8)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=np.unique(y))

if __name__ == "__main__":
    main()
