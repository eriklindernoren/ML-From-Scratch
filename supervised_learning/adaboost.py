from __future__ import division, print_function
import math
import sys
import os
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split
from data_operation import accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

# Decision stump used as weak classifier in Adaboost
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        # List of weak classifiers
        self.clfs = []

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_features))
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Initial values
            err_min = 1
            # Iterate throught every unique sample for each feature
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Try every unique feature value as threshold
                for threshold in unique_values:
                    p = 1
                    err = 0
                    # Set all predictions to '1' initially
                    prediction = np.ones(np.shape(y))
                    # Label the samples whose values are below threshold as '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = missclassified_samples * weights_of_samples
                    err = sum(w[y != prediction])
                    # E.g err = 0.8 => (1 - err) = 0.2
                    # We flip the error and polarity
                    if err > 0.5 and err <= 1:
                        err = 1 - err
                        p = -1
                    # If this threshold resulted in the smallest error we save the
                    # configuration
                    if err < err_min:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        err_min = err
            # Calculate the alpha which is used to update the sample weights
            # and is an approximation of this classifiers proficiency
            clf.alpha = 0.5 * math.log((1.0 - err_min) / (err_min + 0.0001))

            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1

            # Calculate new weights 
            # Missclassified gets larger and correctly classified smaller
            w = w.dot(np.exp(clf.alpha * y.dot(predictions)))
            # Normalize to one
            w /= np.sum(w)
            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        y_pred = np.zeros((np.shape(X)[0], 1))
        # For each classifier label the samples
        for clf in self.clfs:
            pred = np.full((np.shape(X)[0], 1), 1.0)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            pred[negative_idx] = -1.0
            # Add column of predictions weighted by the classifiers alpha
            # (alpha indicative of classifiers profieciency)
            y_pred = np.concatenate((y_pred, clf.alpha * pred), axis=1)
        # Sum weighted predictions and return sign of prediction sum
        y_pred = np.sign(np.sum(y_pred, axis=1))

        return y_pred


def main():
    df = pd.read_csv(dir_path + "/../data/iris.csv")
    # Change class labels from strings to numbers
    df = df.replace(to_replace="setosa", value="-1")
    df = df.replace(to_replace="virginica", value="1")
    df = df.replace(to_replace="versicolor", value="2")

    # Only select data for two classes
    X = df.loc[df['species'] != "2"].drop("species", axis=1).as_matrix()
    y = df.loc[df['species'] != "2"]["species"].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Adaboost classification
    clf = Adaboost(n_clf=8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy:", accuracy_score(y_test, y_pred))

    # Reduce dimensions to 2d using pca and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__":
    main()
