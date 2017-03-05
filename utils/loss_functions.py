from __future__ import division
import numpy as np


class SquareLoss():
    def __init__(self): pass 

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)

    def gradient(self, y_true, y_pred):
        return -1 * (y_true - y_pred)


class LogisticLoss():
    def __init__(self): pass 

    def log_func(self, t, dt=False):
        if dt:
            return self.log_func(t) * (1 - self.log_func(t))
        else:
            return 1 / (1 + np.exp(-t))

    def loss(self, y_true, y_pred):
        return (1/len(y_true)) * (-y_true.T.dot(self.log_func(y_pred) - (1 - y_true.T).dot(self.log_func(y_pred))))

    def gradient(self, y_true, y_pred):
        return self.log_func(y_pred) - y_true
