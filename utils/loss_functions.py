from __future__ import division
import numpy as np


class SquareLoss():
    def __init__(self): pass 

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)

    # W.r.t y_pred
    def gradient(self, y, y_pred):
        return -1 * (y - y_pred)

    def hess(self, y, y_pred):
        return np.ones(np.shape(y))


class LogisticLoss():
    def __init__(self): pass 

    def log_func(self, t, dt=False):
        if dt:
            return self.log_func(t) * (1 - self.log_func(t))
        else:
            return 1 / (1 + np.exp(-t))

    def loss(self, y, y_pred):
        l = -y * self.log_func(y_pred)
        r = -(1 - y) * self.log_func(y_pred)
        loss = (1/len(y)) * (l + r)
        return loss

    def gradient(self, y, y_pred):
        return -(y - self.log_func(y_pred))

    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return -p * (1 - p)

