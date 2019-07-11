from __future__ import division
import math
import numpy as np
from mlfromscratch.utils import accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class MdnLoss(Loss):
    """MeanNegLogLike loss for MDN networks

    """
    def __init__(self, num_components, output_dim):
        self.mixtures = num_components
        self.output = output_dim
        self.eps = 1e-5
        self.ypred = None

    def softmax(self, x):
        z = x - np.max(x, axis=1, keepdims=True)  # numerical stability
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def gaussian_pdf(self, x, mu, sigma):
        denominator = np.sqrt(math.pi * 2) * sigma + self.eps
        numerator = np.exp(-((x - mu)**2) / (2 * sigma**2))
        kernel = numerator / denominator
        return kernel

    # compute loss func(pi, sigma, mu, y_true)
    def loss(self, y_true, y_pred):
        components = y_pred.reshape(-1, 2 + self.output, self.mixtures)
        pi = components[:, :self.output, :]
        pi = np.reshape(pi, (-1, np.prod(pi.shape[1:])))
        mu = components[:, self.output, :]
        sigma = components[:, self.output + 1, :]

        # calculate loss
        result = self.gaussian_pdf(y_true, mu, sigma) * pi
        result = np.sum(result, axis=1, keepdims=True)
        result = -np.log(result + self.eps)
        self.ypred = result
        loss = np.mean(result)
        return loss

    def acc(self, y_true, y_pred):
        # return self.logprob
        return accuracy_score(y_true, self.ypred)

    def gradient(self, y_true, y_pred):
        components = y_pred.reshape(-1, 2 + self.output, self.mixtures)
        N = components.shape[0]  # num of data points
        pi = components[:, :self.output, :]
        pi = np.reshape(pi, (-1, np.prod(pi.shape[1:])))
        # pi = np.clip(pi, self.eps, 1.0)
        mu = components[:, self.output, :]
        sigma = components[:, self.output + 1, :]

        g = self.gaussian_pdf(y_true, mu, sigma) * pi
        normalize = np.sum(g, axis=1, keepdims=True)
        gamma = g / normalize
        dmu = gamma * ((mu - y_true)/np.square(sigma))
        dmu = dmu / N
        dsigma = gamma * (1.0 - np.square(y_true - mu) / (sigma**2))
        dsigma = dsigma / N
        dpi = (pi - gamma) / N
        return dpi, dmu, dsigma
