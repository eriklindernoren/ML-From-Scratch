from __future__ import print_function
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, normalize
from mlfromscratch.utils.data_operation import euclidean_distance, accuracy_score
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils import Plot


class KNN():
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, k=5):
        self.k = k

    # Do a majority vote among the neighbors
    def _majority_vote(self, neighbors, classes):
        max_count = 0
        most_common = None
        # Count class occurences among neighbors
        for c in np.unique(classes):
            # Count number of neighbors with class c
            count = len(neighbors[neighbors[:, 1] == c])
            if count > max_count:
                max_count = count
                most_common = c
        return most_common

    def predict(self, X_test, X_train, y_train):
        classes = np.unique(y_train)
        y_pred = []
        # Determine the class of each sample
        for test_sample in X_test:
            neighbors = []
            # Calculate the distance form each observed sample to the
            # sample we wish to predict
            for j, observed_sample in enumerate(X_train):
                distance = euclidean_distance(test_sample, observed_sample)
                label = y_train[j]
                # Add neighbor information
                neighbors.append([distance, label])
            neighbors = np.array(neighbors)
            # Sort the list of observed samples from lowest to highest distance
            # and select the k first
            k_nearest_neighbors = neighbors[neighbors[:, 0].argsort()][:self.k]
            # Do a majority vote among the k neighbors and set prediction as the
            # class receing the most votes
            label = self._majority_vote(k_nearest_neighbors, classes)
            y_pred.append(label)
        return np.array(y_pred)


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
