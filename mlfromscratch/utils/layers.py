
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

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    """A fully-connected NN layer. 

    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
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

    def output_shape(self):
        return (self.n_units,)


class Conv2D(Layer):
    """A 2D Convolution Layer.

    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, height, width, channels)
        Only needs to be specified for first layer in the network.
    padding: int
        The zero padding that will be added to the input image.
    stride: int
        The stride length of the filters during the convolution over the input.
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
        # Turn image shape into column shape 
        # (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))

        # Calculate output
        output = self.W_col.dot(self.X_col) + self.wb
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return output.transpose(3,0,1,2)

    def backward_pass(self, acc_grad):
        # Reshape accumulated gradient into column shape
        acc_grad = acc_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        
        # Take dot product between column shaped accum. gradient and column shape
        # layer input to determine the gradient at the layer with respect to layer weights
        grad_w = acc_grad.dot(self.X_col.T).reshape(self.W.shape)
        # The gradient with respect to bias terms is the sum similarly to in Dense layer
        grad_wb = np.sum(acc_grad, axis=1, keepdims=True)

        # Update the layers weights
        self.W = self.W_opt.update(self.W, grad_w)
        self.wb = self.wb_opt.update(self.wb, grad_wb)

        # Recalculate the gradient which will be propogated back to prev. layer
        acc_grad = self.W_col.T.dot(acc_grad)
        # Reshape from column shape to image shape
        acc_grad = column_to_image(acc_grad, self.layer_input.shape, self.filter_shape, self.stride, self.padding)

        return acc_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        output_height = (height + 2 * self.padding - self.filter_shape[0]) / self.stride + 1
        output_width = (width + 2 * self.padding - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class PoolingLayer(Layer):
    """A parent class of MaxPooling2D and AveragePooling2D
    """
    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

    def forward_pass(self, X, training=True):
        self.layer_input = X

        batch_size, channels, height, width = X.shape

        _, out_height, out_width = self.output_shape()

        X = X.reshape(batch_size*channels, 1, height, width)
        X_col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        
        # MaxPool or AveragePool specific method
        output = self._pool_forward(X_col)

        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)

        return output

    def backward_pass(self, acc_grad):
        batch_size, _, _, _ = acc_grad.shape
        channels, height, width = self.input_shape
        acc_grad = acc_grad.transpose(2, 3, 0, 1).ravel()

        # MaxPool or AveragePool specific method
        acc_grad_col = self._pool_backward(acc_grad)

        acc_grad = column_to_image(acc_grad_col, (batch_size * channels, 1, height, width), self.pool_shape, self.stride, 0)
        acc_grad = acc_grad.reshape((batch_size,) + self.input_shape)

        return acc_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, acc_grad):
        acc_grad_col = np.zeros((np.prod(self.pool_shape), acc_grad.size))
        arg_max = self.cache
        acc_grad_col[arg_max, range(acc_grad.size)] = acc_grad
        return acc_grad_col

class AveragePooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        output = np.mean(X_col, axis=0)
        return output

    def _pool_backward(self, acc_grad):
        acc_grad_col = np.zeros((np.prod(self.pool_shape), acc_grad.size))
        acc_grad_col[:, range(acc_grad.size)] = 1. / acc_grad_col.shape[0] * acc_grad
        return acc_grad_col


class ConstantPadding2D(Layer):
    """ Add constant value to borders of image shaped input (batch_size, channels, height, width) """
    def __init__(self, padding, padding_value=0):
        self.padding = padding
        if isinstance(padding[0], int):
            self.padding = ((padding[0], padding[0]), padding[1])
        if isinstance(padding[1], int):
            self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = padding_value

    def forward_pass(self, X, training=True):
        output = np.pad(X, 
            pad_width=((0,0), (0,0), self.padding[0], self.padding[1]), 
            mode="constant",
            constant_values=self.padding_value)
        return output

    def backward_pass(self, acc_grad):
        height = self.input_shape[1]
        width = self.input_shape[2]
        acc_grad = acc_grad[:, :, self.padding[0][0]:height+self.padding[0][1], self.padding[1][0]:width+self.padding[1][1]]
        return acc_grad

    def output_shape(self):
        new_height = self.input_shape[1] + np.sum(self.padding[0])
        new_width = self.input_shape[2] + np.sum(self.padding[1])
        return (self.input_shape[0], new_height, new_width)

class ZeroPadding2D(ConstantPadding2D):
    """ Add zeros to borders of image shaped input (batch_size, channels, height, width) """
    def __init__(self, padding):
        self.padding = padding
        if isinstance(padding[0], int):
            self.padding = ((padding[0], padding[0]), padding[1])
        if isinstance(padding[1], int):
            self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = 0

class Flatten(Layer):
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self):
        self.prev_shape = None

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, acc_grad):
        return acc_grad.reshape(self.prev_shape)

    def output_shape(self):
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

    def output_shape(self):
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

    def output_shape(self):
        return self.input_shape



# Reference: CS231n Stanford
def get_im2col_indices(images_shape, filter_shape, padding=1, stride=1):
  # First figure out what the size of the output should be
  batch_size, channels, height, width = images_shape
  filter_height, filter_width = filter_shape
  assert (height + 2 * padding - filter_height) % stride == 0
  assert (width + 2 * padding - filter_height) % stride == 0
  out_height = (height + 2 * padding - filter_height) / stride + 1
  out_width = (width + 2 * padding - filter_width) / stride + 1

  i0 = np.repeat(np.arange(filter_height), filter_width)
  i0 = np.tile(i0, channels)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(filter_width), filter_height * channels)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

  return (k, i, j)



# Reference: CS231n Stanford
def image_to_column(images, filter_shape, stride, padding):
    filter_height, filter_width = filter_shape
    p = padding
    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, padding, stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols



# Reference: CS231n Stanford
def column_to_image(cols, images_shape, filter_shape, stride, padding):
    batch_size, channels, height, width = images_shape
    p = padding
    height_padded, width_padded = height + 2 * p, width + 2 * p
    images_padded = np.empty((batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, padding, stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, p:height+p, p:width+p]

