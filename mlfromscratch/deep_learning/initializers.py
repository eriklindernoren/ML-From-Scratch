# -*- coding: utf-8 -*-

import numpy as np

class WeightInitializer:

    def compute_fans(self, shape):
        # Original code forked from MIT licensed keras project
        # https://github.com/fchollet/keras/blob/master/keras/initializers.py
        # kernel shape: ('NF': Total Filters, 'CF': Filter Channels, 'HF': Filter Height 'WF': Filter Width)

        shape = (shape[0], 1) if len(shape) ==  1 else shape
        receptive_field_size = np.prod(shape[:2])
        fan_out = shape[0] * receptive_field_size # NF *receptive_field_size
        fan_in = shape[1] * receptive_field_size # CF *receptive_field_size
        return fan_in, fan_out


class HeNormal(WeightInitializer):

    """
    **He Normal (HeNormal)**

    HeNormal is a robust initialization method that particularly considers the
    rectifier nonlinearities. He normal is an implementation based on
    Gaussian distribution

    References:
        [1] Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            * [Kaiming He, 2015] https://arxiv.org/abs/1502.01852
            * [PDF] https://arxiv.org/pdf/1502.01852.pdf

        [2] Initialization Of Deep Networks Case of Rectifiers
            [DeepGrid Article - Jefkine Kafunah] https://goo.gl/TBNw5t
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(2. / fan_in)
        return np.random.normal(loc = 0.0, scale = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class HeUniform(WeightInitializer):

    """
    **He Normal (HeNormal)**

    HeNormal is a robust initialization method that particularly considers the
    rectifier nonlinearities. He uniform is an implementation based on
    Uniform distribution

    References:
        [1] Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            * [Kaiming He, 2015] https://arxiv.org/abs/1502.01852
            * [PDF] https://arxiv.org/pdf/1502.01852.pdf

        [2] Initialization Of Deep Networks Case of Rectifiers
            [DeepGrid Article - Jefkine Kafunah] https://goo.gl/TBNw5t
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(6. / fan_in)
        return np.random.uniform(low = -scale, high = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class GlorotNormal(WeightInitializer):

    """
    **Glorot Normal (GlorotNormal)**

    GlorotNormal more famously known as the Xavier initialization is based on the
    effort to try mantain the same variance of the gradients of the weights for
    all the layers. Glorot normal is an implementation based on
    Gaussian distribution

    References:
        [1] Understanding the difficulty of training deep feedforward neural networks
            * [Xavier Glorot, 2010] http://proceedings.mlr.press/v9/glorot10a.html
            * [PDF] http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        [2] Initialization Of Deep Feedfoward Networks
            [DeepGrid Article - Jefkine Kafunah] https://goo.gl/E2XrGe
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(2. / (fan_in + fan_out))
        return np.random.normal(loc = 0.0, scale = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class GlorotUniform(WeightInitializer):

    """
    **Glorot Uniform (GlorotUniform)**

    GlorotUniform more famously known as the Xavier initialization is based on the
    effort to try mantain the same variance of the gradients of the weights for
    all the layers. Glorot uniform is an implementation based on
    Uniform distribution

    References:
        [1] Understanding the difficulty of training deep feedforward neural networks
            * [Xavier Glorot, 2010] http://proceedings.mlr.press/v9/glorot10a.html
            * [PDF] http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        [2] Initialization Of Deep Feedfoward Networks
            [DeepGrid Article - Jefkine Kafunah] https://goo.gl/E2XrGe
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(6. / (fan_in + fan_out))
        return np.random.uniform(low = -scale, high = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class LeCunUniform(WeightInitializer):

    """
    **LeCun Uniform (LeCunUniform)**

    Weights should be randomly chosen but in such a way that the sigmoid is
    primarily activated in its linear region. LeCun uniform is an
    implementation based on Uniform distribution

    References:
        [1] Efficient Backprop
            [LeCun, 1998][PDF] http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(3. / fan_in)
        return np.random.uniform(low = -scale, high = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class LeCunNormal(WeightInitializer):

    """
    **LeCun Normal (LeCunNormal)**

    Weights should be randomly chosen but in such a way that the sigmoid is
    primarily activated in its linear region. LeCun uniform is the
    implementation based on Gaussian distribution

    References:
        [1] Efficient Backprop
            [LeCun, 1998][PDF] http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """

    def weights(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(1. / fan_in)
        return np.random.normal(low = -scale, high = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class RandomUniform(WeightInitializer):

    """
    **Random Uniform (RandomUniform)**

    Random uniform is an implementation of weight initialization based on
    Uniform distribution
    """

    def weights(self, shape, seed = None):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(1. / (fan_in + fan_out))
        np.random.seed(seed)
        return np.random.uniform(low = -scale, high = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class RandomNormal(WeightInitializer):

    """
    **Random Normal (RandomNormal)**

    Random uniform is an implementation of weight initialization based on
    Gaussian distribution
    """

    def weights(self, shape, seed = None):
        fan_in, fan_out = self.compute_fans(shape)
        scale = np.sqrt(1. / (fan_in + fan_out))
        np.random.seed(seed)
        return np.random.normal(loc = 0.0, scale = scale, size = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class Zero(WeightInitializer):

    """
    **Zero (Zero)**

    Zero is an implementation of weight initialization that returns all zeros
    """

    def weights(self, shape):
        return np.zeros(shape = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class One(WeightInitializer):

    """
    **One (One)**

    One is an implementation of weight initialization that returns all ones
    """

    def weights(self, shape):
        return np.ones(shape = shape)

    @property
    def init_name(self):
        return self.__class__.__name__


class InitializeWeights:

    _methods = {
        'ones': One,
        'zeros': Zero,
        'he-normal': HeNormal,
        'he-uniform': HeUniform,
        'lecun-normal': LeCunNormal,
        'lecun-uniform': LeCunUniform,
        'random-normal': RandomNormal,
        'glorot-normal': GlorotNormal,
        'random-uniform': RandomUniform,
        'glorot-uniform': GlorotUniform
    }

    def __init__(self, name):
        if name not in self._methods.keys():
            raise Exception('Weight initialization method must be either one of the following: {}.'.format(', '.join(self._methods.keys())))
        self.init_method = self._methods[name]()

    @property
    def name(self):
        return self.init_method.init_name

    def initialize_weights(self, shape):
        return self.init_method.weights(shape)
