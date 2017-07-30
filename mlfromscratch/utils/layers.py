
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


class Dense(Layer):
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
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.wb = np.zeros((1, self.n_units))
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
    padding: int
        The zero padding that will be added to the input image.
    stride: int
        The stride length of the filters during the convolution over the input.
    activation_function: class:
        The activation function that will be used for each unit. 
        Possible choices: Sigmoid, ELU, ReLU, LeakyReLU, SoftPlus, TanH, SELU, Softmax
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding=0, stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape

    def initialize(self, optimizer):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W  = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.wb = np.zeros((self.n_filters, 1))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.wb_opt = copy.copy(optimizer)

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self.W.reshape((self.n_filters, -1))

        output = self.col_W.dot(self.col) + self.wb
        output = output.reshape(self.shape() + (batch_size, ))
        return output.transpose(3,0,1,2)

    def backward_pass(self, acc_grad):
        acc_grad = acc_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        
        grad_w = acc_grad.dot(self.col.T).reshape(self.W.shape)
        grad_wb = np.sum(acc_grad, axis=1, keepdims=True)

        self.W = self.W_opt.update(self.W, grad_w)
        self.wb = self.wb_opt.update(self.wb, grad_wb)

        acc_grad = self.col_W.T.dot(acc_grad)
        acc_grad = column_to_image(acc_grad, self.layer_input.shape, self.filter_shape, self.stride, self.padding)

        return acc_grad

    def shape(self):
        height, width = convolution_shape(self.input_shape, self.filter_shape, self.stride, self.padding)
        return self.n_filters, height, width

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

class Dropout(Layer):
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


class Activation(Layer):

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

    def __init__(self, name):
        self.activation = self.activation_functions[name]()

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation.function(X)

    def backward_pass(self, acc_grad):
        return acc_grad * self.activation.gradient(self.layer_input)

    def shape(self):
        return self.input_shape

def get_im2col_indices(x_shape, filter_shape, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  filter_height, filter_width = filter_shape
  assert (H + 2 * padding - filter_height) % stride == 0
  assert (W + 2 * padding - filter_height) % stride == 0
  out_height = (H + 2 * padding - filter_height) / stride + 1
  out_width = (W + 2 * padding - filter_width) / stride + 1

  i0 = np.repeat(np.arange(filter_height), filter_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(filter_width), filter_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)

  return (k, i, j)

# Reference: CS231n Stanford
def image_to_column(X, filter_shape, stride, padding):
    filter_height, filter_width = filter_shape
    p = padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(X.shape, filter_shape, padding, stride)

    cols = X_padded[:, k, i, j]
    channels = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols

# Reference: CS231n Stanford
def column_to_image(cols, images_shape, filter_shape, stride, padding):
    batch_size, channels, height, width = images_shape
    p = padding
    height_padded, width_padded = height + 2 * p, width + 2 * p
    X_padded = np.empty((batch_size, channels, height_padded, width_padded))
    k, i, j = get_im2col_indices(images_shape, filter_shape, padding, stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    np.add.at(X_padded, (slice(None), k, i, j), cols)

    return X_padded[:, :, p:height+p, p:width+p]

def convolution_shape(input_shape, filter_shape, stride, padding):
    """Calculate output height and width for a convolution layer."""
    _, img_height, img_width = input_shape
    height = (img_height + 2 * padding - filter_shape[0]) / float(stride) + 1
    width = (img_width + 2 * padding - filter_shape[1]) / float(stride) + 1

    return int(height), int(width)
