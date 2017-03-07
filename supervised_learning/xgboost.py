from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, standardize, categorical_to_binary, normalize
from data_operation import mean_squared_error, accuracy_score
from decision_tree import XGBoostRegressionTree
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class LogisticLoss():
    def __init__(self): pass 

    def log_func(self, t, dt=False):
        if dt:
            return self.log_func(t) * (1 - self.log_func(t))
        else:
            return 1 / (1 + np.exp(-t))

    def gradient(self, y, y_pred):
        return y * (y - self.log_func(y_pred))

    def hess(self, y, y_pred):
        prob = self.log_func(y_pred)
        return prob * (1 - prob)


# Super class to GradientBoostingRegressor and GradientBoostingClassifier
class XGBoost(object):
    def __init__(self, n_estimators=20, learning_rate=0.5, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), debug=False):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.init_estimate = None                   # The initial prediction of y
        self.debug = debug
        
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
        y = categorical_to_binary(y)

        # Set initial predictions to median of y
        self.init_estimate = np.median(y, axis=0)
        y_pred = np.zeros(np.shape(y))
        for i, tree in enumerate(self.trees):
            gradient = self.loss.gradient(y, y_pred)
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)

            gradient_est = tree.predict(X)

            # Update y prediction by the gradient value
            y_pred += np.multiply(self.learning_rate, gradient_est)

            progress = 100 * (i / self.n_estimators)
            if self.debug:
                print ("Progress: %.2f%%" % progress)

    def predict(self, X):
        # Fix shape of y_pred as (n_samples, n_outputs)
        n_samples = np.shape(X)[0]
        if not np.shape(self.init_estimate):
            y_pred = np.full(n_samples, self.init_estimate)
        else:
            n_outputs = np.shape(self.init_estimate)[0]
            y_pred = np.full((n_samples, n_outputs), self.init_estimate)
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            gradient_est = tree.predict(X)
            gradient_est = np.array(gradient_est).reshape(np.shape(y_pred))
            y_pred += np.multiply(self.learning_rate, gradient_est)

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

    print ("Accuracy:", accuracy_score(y_test, y_pred))

    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__":
    main()