from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize, categorical_to_binary, normalize
from mlfromscratch.utils.data_operation import mean_squared_error, accuracy_score
from mlfromscratch.supervised_learning import XGBoostRegressionTree
from mlfromscratch.utils.loss_functions import LogisticLoss
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.utils import Plot


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
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.debug = debug

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        # Log loss for classification
        self.loss = LogisticLoss(grad_wrt_theta=False)

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
        y = categorical_to_binary(y)

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

        # Turn into probability distribution
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

def main():

    print ("-- XGBoost --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)  

    clf = XGBoost(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, 
        title="XGBoost", 
    accuracy=accuracy, 
    legend_labels=data.target_names)


if __name__ == "__main__":
    main()