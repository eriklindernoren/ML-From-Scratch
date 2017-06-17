from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import line_search
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize, categorical_to_binary
from mlfromscratch.utils.data_operation import mean_squared_error, accuracy_score
from mlfromscratch.utils.loss_functions import SquareLoss, LogisticLoss
from mlfromscratch.supervised_learning.decision_tree import RegressionTree
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.utils import Plot


# Super class to GradientBoostingRegressor and GradientBoostingClassifier
class GradientBoosting(object):
    """Super class of GradientBoostingClassifier and GradientBoostinRegressor. 
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function. 

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
    regression: boolean
        True or false depending on if we're doing regression or classification.
    debug: boolean
        True or false depending on if we wish to display the training progress.
    """
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression, debug):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.init_estimate = None
        self.regression = regression
        self.debug = debug
        self.multipliers = []
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        # Square loss for regression
        # Log loss for classification
        self.loss = SquareLoss(grad_wrt_theta=False)
        if not self.regression:
            self.loss = LogisticLoss(grad_wrt_theta=False)

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth)
            self.trees.append(tree)


    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        
        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            gradient = self.loss.gradient(y, y_pred)
            tree.fit(X, gradient)
            update = tree.predict(X)
            # Update y prediction
            y_pred -= np.multiply(self.learning_rate, update)


    def predict(self, X):
        y_pred = np.array([])
        # Make predictions
        for i, tree in enumerate(self.trees):
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            # prediction = np.array(prediction).reshape(np.shape(y_pred))
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True,
            debug=debug)

class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False,
            debug=debug)

    def fit(self, X, y):
        y = categorical_to_binary(y)
        super(GradientBoostingClassifier, self).fit(X, y)


def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = GradientBoostingClassifier(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


    Plot().plot_in_2d(X_test, y_pred, 
        title="Gradient Boosting", 
        accuracy=accuracy, 
        legend_labels=data.target_names)

    print ("-- Gradient Boosting Regression --")

    X, y = datasets.make_regression(n_features=1, n_samples=150, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.5)

    clf = GradientBoostingRegressor(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.title("Gradient Boosting Regression (%.2f MSE)" % mse)
    plt.show()


if __name__ == "__main__":
    main()