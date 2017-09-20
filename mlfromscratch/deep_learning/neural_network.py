from __future__ import print_function
from terminaltables import AsciiTable
import numpy as np
import progressbar
from mlfromscratch.utils import batch_iterator
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets


class NeuralNetwork():
    """Neural Network. Deep Learning base model.

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
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        # Add layer to the network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        # Calculate the loss and accuracy of the prediction
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)

        bar = progressbar.ProgressBar(widgets=bar_widgets)
        for _ in bar(range(n_epochs)):
            batch_error = 0
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error += loss

            self.errors["training"].append(batch_error / n_batches)

            if self.validation_set:
                # Determine validation error
                y_val_pred = self._forward_pass(self.X_val)
                validation_loss = np.mean(self.loss_function.loss(self.y_val, y_val_pred))
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
        # Iterate through network and get each layer's configuration
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
