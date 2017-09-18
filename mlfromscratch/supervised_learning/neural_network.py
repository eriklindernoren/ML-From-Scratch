from __future__ import print_function
from sklearn import datasets
from terminaltables import AsciiTable
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
from mlfromscratch.utils.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from mlfromscratch.utils.layers import AveragePooling2D, ZeroPadding2D, BatchNormalization, RNN



class NeuralNetwork():
    """Neural Network.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels
    """
    def __init__(self, optimizer, loss=CrossEntropy, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        
        self.validation_set = False
        if validation_data:
            self.validation_set = True
            self.X_val, self.y_val = validation_data

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If this is not the first layer added then set the input shape
        # as the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to the network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        # Calculate output
        y_pred = self._forward_pass(X)
        # Calculate the training loss
        loss = np.mean(self.loss_function.loss(y, y_pred))
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Calculate the accuracy of the prediction
        acc = self.loss_function.acc(y, y_pred)
        # Backprop. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc


    def fit(self, X, y, n_epochs, batch_size):

        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)

        bar = progressbar.ProgressBar(widgets=bar_widgets)
        for _ in bar(range(n_epochs)):
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
            if self.validation_set:
                # Determine validation error
                y_val_p = self._forward_pass(self.X_val)
                validation_loss = np.mean(self.loss_function.loss(self.y_val, y_val_p))
                self.errors["validation"].append(validation_loss)

        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN. The output of layer l1 becomes the
        input of the following layer l2 """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        """ Propogate the gradient 'backwards' and update the weights
        in each layer. The output of l2 becomes the input of l1. """
        acc_grad = loss_grad
        for layer in reversed(self.layers):
            acc_grad = layer.backward_pass(acc_grad)

    def summary(self, name="Model Summary"):

        # Print model name
        print (AsciiTable([[name]]).table)

        # Network input shape (first layer's input shape)
        print ("Input Shape: %s" % str(self.layers[0].input_shape))

        # Get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params

        # Print network configuration table
        print (AsciiTable(table_data).table)

        print ("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)


def main():

    optimizer = Adam()

    def gen_mult_ser(nums):
        """ Method which generates multiplication series """
        X = np.zeros([nums, 10, 61], dtype=float)
        y = np.zeros([nums, 10, 61], dtype=float)
        for i in range(nums):
            start = np.random.randint(2, 7)
            mult_ser = np.linspace(start, start*10, num=10, dtype=int)
            X[i] = to_categorical(mult_ser, n_col=61)
            y[i] = np.roll(X[i], -1, axis=0)
        y[:, -1, 1] = 1 # Mark endpoint as 1
        return X, y


    def gen_num_seq(nums):
        """ Method which generates sequence of numbers """
        X = np.zeros([nums, 10, 20], dtype=float)
        y = np.zeros([nums, 10, 20], dtype=float)
        for i in range(nums):
            start = np.random.randint(0, 10)
            num_seq = np.arange(start, start+10)
            X[i] = to_categorical(num_seq, n_col=20)
            y[i] = np.roll(X[i], -1, axis=0)
        y[:, -1, 1] = 1 # Mark endpoint as 1
        return X, y

    X, y = gen_mult_ser(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Model definition
    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy)
    clf.add(RNN(10, activation="tanh", bptt_trunc=5, input_shape=(10, 61)))
    clf.add(Activation('softmax'))
    clf.summary("RNN")

    # Print a problem instance and the correct solution
    tmp_X = np.argmax(X_train[0], axis=1)
    tmp_y = np.argmax(y_train[0], axis=1)
    print ("Number Series Problem:")
    print ("X = [" + " ".join(tmp_X.astype("str")) + "]")
    print ("y = [" + " ".join(tmp_y.astype("str")) + "]")
    print ()

    train_err, _ = clf.fit(X_train, y_train, n_epochs=500, batch_size=512)

    # Predict labels of the test data
    y_pred = np.argmax(clf.predict(X_test), axis=2)
    y_test = np.argmax(y_test, axis=2)

    print ()
    print ("Results:")
    for i in range(5):
        # Print a problem instance and the correct solution
        tmp_X = np.argmax(X_test[i], axis=1)
        tmp_y1 = y_test[i]
        tmp_y2 = y_pred[i]
        print ("X      = [" + " ".join(tmp_X.astype("str")) + "]")
        print ("y_true = [" + " ".join(tmp_y1.astype("str")) + "]")
        print ("y_pred = [" + " ".join(tmp_y2.astype("str")) + "]")
        print ()
    
    accuracy = np.mean(accuracy_score(y_test, y_pred))
    print ("Accuracy:", accuracy)

    training = plt.plot(range(500), train_err, label="Training Error")
    plt.title("Error Plot")
    plt.ylabel('Training Error')
    plt.xlabel('Iterations')
    plt.show()


    #-----
    # MLP
    #-----

    # data = datasets.load_digits()
    # X = data.data
    # y = data.target

    # # Convert to one-hot encoding
    # y = to_categorical(y.astype("int"))

    # n_samples = np.shape(X)
    # n_hidden = 512

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

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

    # print ()
    # clf.summary(name="MLP")
    
    # clf.fit(X_train, y_train, n_epochs=50, batch_size=256)
    # clf.plot_errors()

    # y_pred = np.argmax(clf.predict(X_test), axis=1)

    # accuracy = accuracy_score(y_test, y_pred)
    # print ("Accuracy:", accuracy)


    #----------
    # Conv Net
    #----------

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    n_samples = np.shape(X)
    n_hidden = 512

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1,1,8,8))
    X_test = X_test.reshape((-1,1,8,8))

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), input_shape=(1,8,8), padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Conv2D(n_filters=32, filter_shape=(3,3), padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Flatten())
    clf.add(Dense(256))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.5))
    clf.add(BatchNormalization())
    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="ConvNet")

    train_err, val_err = clf.fit(X_train, y_train, n_epochs=50, batch_size=256)
    
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    # Predict labels of the test data
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Flatten data set
    X_test = X_test.reshape(-1, 8*8)

    # Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=range(10))

if __name__ == "__main__":
    main()
