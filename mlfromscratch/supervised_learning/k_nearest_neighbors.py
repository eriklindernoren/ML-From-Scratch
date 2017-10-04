from __future__ import print_function, division
import numpy as np
from mlfromscratch.utils import euclidean_distance

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

    def _vote(self, neighbors):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbors[:, 1].astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Two columns [distance, label], for each observed sample
            neighbors = np.empty((X_train.shape[0], 2))
            # Calculate the distance from each observed sample to the
            # sample we wish to predict
            for j, observed_sample in enumerate(X_train):
                distance = euclidean_distance(test_sample, observed_sample)
                label = y_train[j]
                # Add neighbor information
                neighbors[j] = [distance, label]
            # Sort the list of observed samples from lowest to highest distance
            # and select the k first
            k_nearest_neighbors = neighbors[neighbors[:, 0].argsort()][:self.k]
            # Get the most common class among the neighbors
            label = self._vote(k_nearest_neighbors)
            y_pred[i] = label
        return y_pred
        