from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, standardize, categorical_to_binary
from data_operation import mean_squared_error, accuracy_score
from loss_functions import SquareLoss, LogisticLoss
from decision_tree import RegressionTree
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


# Super class to GradientBoostingRegressor and GradientBoostingClassifier
class GradientBoosting(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.init_estimate = None                   # The initial prediction of y
        self.regression = regression
        
        # Square loss for regression
        # Log loss for classification
        self.loss = SquareLoss()
        if not self.regression:
            self.loss = LogisticLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth)

            self.trees.append(tree)

    def fit(self, X, y):
        # Set initial predictions to median of y
        self.init_estimate = np.median(y, axis=0)
        y_pred = self.init_estimate * np.ones(np.shape(y))
        for tree in self.trees:
            
            gradient = self.loss.gradient(y, y_pred)
            tree.fit(X, gradient)
            gradient_est = tree.predict(X)

            # Make sure shape is same as y_pred
            gradient_est = np.array(gradient_est).reshape(np.shape(y_pred))

            # Update y prediction by the estimated gradient value
            y_pred -= np.multiply(self.learning_rate, gradient_est)

    def predict(self, X):
        # Fix shape of y_pred as (n_samples, n_outputs)
        n_samples = np.shape(X)[0]
        if not np.shape(self.init_estimate):
            y_pred = self.init_estimate * np.ones((n_samples, ))
        else:
            y_pred = self.init_estimate * np.ones((n_samples, np.shape(self.init_estimate)[0]))

        # Make predictions
        for tree in self.trees:
            prediction = tree.predict(X)
            prediction = np.array(prediction).reshape(np.shape(y_pred))
            y_pred -= np.multiply(self.learning_rate, prediction)

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=20, learning_rate=.8, min_samples_split=20,
                 min_var_red=1e-4, max_depth=20):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True)

class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=20, learning_rate=1, min_samples_split=20,
                 min_info_gain=1e-7, max_depth=20):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)

    def fit(self, X, y):
        y = categorical_to_binary(y)
        super(GradientBoostingClassifier, self).fit(X, y)


def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy:", accuracy_score(y_test, y_pred))

    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)

    print ("-- Gradient Boosting Regression --")

    X, y = datasets.make_regression(n_features=1, n_samples=100, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.5)

    clf = GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    print ("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.show()


if __name__ == "__main__":
    main()