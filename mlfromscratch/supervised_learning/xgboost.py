from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize, to_categorical, normalize
from mlfromscratch.utils.data_operation import mean_squared_error, accuracy_score
from mlfromscratch.supervised_learning import XGBoostRegressionTree
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils.activation_functions import Sigmoid
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.utils import Plot


class LogisticLoss():
    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid.function
        self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    # gradient w.r.t y_pred
    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    # w.r.t y_pred
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)


class XGBoost(object):
    """The XGBoost classifier.

    Reference: http://xgboost.readthedocs.io/en/latest/model.html

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further. 
    max_depth: int
        The maximum depth of a tree.
    debug: boolean
        True or false depending on if we wish to display the training progress.
    """
    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, debug=False):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate          # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.debug = debug

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        # Log loss for classification
        self.loss = LogisticLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        y = to_categorical(y)

        y_pred = np.zeros(np.shape(y))

        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)

            y_pred -= np.multiply(self.learning_rate, update_pred)

    def predict(self, X):
        # Fix shape of y_pred as (n_samples, n_outputs)
        n_samples = np.shape(X)[0]
        y_pred = np.array([])
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update = np.multiply(self.learning_rate, tree.predict(X))
            y_pred = update if not y_pred.any() else y_pred - update

        # Turn into probability distribution (Softmax)
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
