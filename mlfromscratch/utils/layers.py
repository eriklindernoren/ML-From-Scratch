
from __future__ import print_function
import sys
import os
import math
import numpy as np
import copy
from mlfromscratch.utils.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU, Softmax


class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, acc_grad):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


class DenseLayer(Layer):
    """A fully-connected NN layer. 

    Parameters:
    -----------
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    n_units: int
        The number of neurons in the layer.
    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.initialized = False
        self.input_shape, self.n_units = input_shape, n_units
        self.W = None
        self.wb = None

    def initialize(self, optimizer):
        # Initialize the weights
        a, b = -1 / math.sqrt(self.input_shape[0]), 1 / math.sqrt(self.input_shape[0])
        self.W  = (b - a) * np.random.random((self.input_shape[0], self.n_units)) + a
        self.wb = (b - a) * np.random.random((1, self.n_units)) + a
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.wb

    def backward_pass(self, acc_grad):
        # Calculate gradient w.r.t layer weights
        grad_w = self.layer_input.T.dot(acc_grad)
        grad_wb = np.sum(acc_grad, axis=0, keepdims=True)

        # Update the layer weights
        self.W = self.W_opt.update(self.W, grad_w)
        self.wb = self.wb_opt.update(self.wb, grad_wb)

        # Return accumulated gradient for next layer
        acc_grad = acc_grad.dot(self.W.T)
        return acc_grad

    def shape(self):
        return (self.n_units,)


class Conv2D(Layer):
    """A 2D Convolution Layer.

    Parameters:
    -----------
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, height, width, channels) 
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    padding: tuple
        The zeros padding that will be added to the input image.
    stride: int
        The stride length of the filters during the convolution over the input.
    activation_function: class:
        The activation function that will be used for each unit. 
        Possible choices: Sigmoid, ELU, ReLU, LeakyReLU, SoftPlus, TanH, SELU, Softmax
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding=(0, 0), stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape

    def initialize(self, optimizer):
        # Initialize the weights
        self.W  = np.random.random((self.n_filters, self.filter_shape[0], self.filter_shape[1], self.input_shape[-1]))
        self.wb = np.random.random((1, self.n_filters))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def forward_pass(self, X, training=True):
        batch_size, height, width, n_filters = X.shape
        self.layer_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self.W.reshape((self.n_filters, -1)).T

        output = (self.col.dot(self.col_W) + self.wb).reshape((batch_size, height, width, -1))
        return output

    def backward_pass(self, acc_grad):
        acc_grad = acc_grad.reshape((self.n_filters, -1)).T
        
        grad_w = self.col.T.dot(acc_grad).reshape(self.W.shape)
        grad_wb = np.sum(acc_grad, axis=0, keepdims=True)

        self.W = self.W_opt.update(self.W, grad_w)
        self.wb = self.wb_opt.update(self.wb, grad_wb)

        acc_grad = acc_grad.dot(self.col_W.T)
        acc_grad = column_to_image(acc_grad, self.layer_input.shape, self.filter_shape, self.stride, self.padding)

        return acc_grad

    def shape(self):
        return convolution_shape(self.input_shape, self.n_filters, self.filter_shape, self.stride, self.padding)

class Flatten(Layer):
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self):
        self.prev_shape = None

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, acc_grad):
        return acc_grad.reshape(self.prev_shape)

    def shape(self):
        return (np.prod(self.input_shape),)

class DropoutLayer(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p: float
        The probability that unit x is set to zero.
    """
    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True

    def forward_pass(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, acc_grad):
        return acc_grad * self._mask

    def shape(self):
        return self.input_shape


activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'softplus': SoftPlus
}

class Activation(Layer):
    def __init__(self, name):
        self.activation = activation_functions[name]()

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation.function(X)

    def backward_pass(self, acc_grad):
        return acc_grad * self.activation.gradient(self.layer_input)

    def shape(self):
        return self.input_shape

def image_to_column(images, filter_shape, stride, padding):
    batch_size, height, width, channels = images.shape
    f_height, f_width = filter_shape
    out_height, out_width, _ = convolution_shape(images.shape[1:], filter_shape, stride, padding)
    images = np.pad(images, ((0, 0), padding, padding, (0, 0)), mode='constant')

    col = np.zeros((batch_size, f_height, f_width, out_height, out_width, channels))
    for y in range(f_height):
        y_bound = y + stride * out_height
        for x in range(f_width):
            x_bound = x + stride * out_width
            col[:, y, x, :, :, :] = images[:, y:y_bound:stride, x:x_bound:stride, :]

    col = col.reshape(batch_size * out_height * out_width, -1)
    return col


def column_to_image(columns, images_shape, filter_shape, stride, padding):
    n_images, height, width, channels = images_shape
    f_height, f_width = filter_shape

    out_height, out_width, _ = convolution_shape(images_shape[1:], filter_shape, stride, padding)
    columns = columns.reshape(n_images, out_height, out_width, f_height, f_width, channels).transpose(0, 3, 4, 1,
                                                                                                        2, 5)

    img_h = height + 2 * padding[0] + stride - 1
    img_w = width + 2 * padding[1] + stride - 1
    img = np.zeros((n_images, img_h, img_w, channels))
    for y in range(f_height):
        y_bound = y + stride * out_height
        for x in range(f_width):
            x_bound = x + stride * out_width
            img[:, y:y_bound:stride, x:x_bound:stride, :] += columns[:, y, x, :, :, :]

    return img[:, padding[0]:height + padding[0], padding[1]:width + padding[1], :]

def convolution_shape(input_shape, n_filters, filter_shape, stride, padding):
    """Calculate output shape for a convolution layer."""
    img_height, img_width, _ = input_shape
    height = (img_height + 2 * padding[0] - filter_shape[0]) / float(stride) + 1
    width = (img_width + 2 * padding[1] - filter_shape[1]) / float(stride) + 1

    return int(height), int(width), n_filters
