from __future__ import division
import numpy as np
from activation_functions import Sigmoid


class SquareLoss():
    def __init__(self, grad_wrt_theta=True):
        if grad_wrt_theta:
            self.gradient = self._grad_wrt_theta
        if not grad_wrt_theta:
            self.gradient = self._grad_wrt_pred

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)

    def _grad_wrt_pred(self, y, y_pred):
        return -(y - y_pred)

    def _grad_wrt_theta(self, y, X, theta):
        y_pred = X.dot(theta)
        return -(y - y_pred).dot(X)


class CrossEntropy():
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class LogisticLoss():
    def __init__(self, grad_wrt_theta=True):
        sigmoid = Sigmoid()
        self.log_func = sigmoid.function
        self.log_grad = sigmoid.gradient

        if grad_wrt_theta:
            self.gradient = self._grad_wrt_theta
        if not grad_wrt_theta:
            self.gradient = self._grad_wrt_pred
            self.hess = self._hess_wrt_pred

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    # gradient w.r.t theta
    def _grad_wrt_theta(self, y, X, theta):
        p = self.log_func(X.dot(theta))
        return -(y - p).dot(X)

    # gradient w.r.t y_pred
    def _grad_wrt_pred(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    # w.r.t y_pred
    def _hess_wrt_pred(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)