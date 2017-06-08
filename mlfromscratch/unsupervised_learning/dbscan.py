import sys
import os
import math
import random
from sklearn import datasets
import numpy as np

# Import helper functions
from mlfromscratch.utils.data_manipulation import normalize
from mlfromscratch.utils.data_operation import euclidean_distance
from mlfromscratch.unsupervised_learning import PCA


class DBSCAN():
    """A density based clustering method that expands clusters from 
    samples that have more neighbors within a radius specified by eps
    than the value min_samples.

    Parameters:
    -----------
    eps: float
        The radius within which samples are considered neighbors
    min_samples: int
        The number of neighbors required for the sample to be a core point. 
    """
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        # List of arrays (clusters) containing sample indices
        self.clusters = []
        self.visited_samples = []
        # Hashmap {"sample_index": [neighbor1, neighbor2, ...]}
        self.neighbors = {}
        self.X = None   # Dataset

    # Return a list of neighboring samples
    # A sample_2 is considered a neighbor of sample_1 if the distance between
    # them is smaller than epsilon
    def _get_neighbors(self, sample_i):
        neighbors = []
        for _sample_i, _sample in enumerate(self.X):
            if _sample_i != sample_i and euclidean_distance(
                    self.X[sample_i], _sample) < self.eps:
                neighbors.append(_sample_i)
        return np.array(neighbors)

    # Recursive method which expands the cluster until we have reached the border
    # of the dense area (density determined by eps and min_samples)
    def _expand_cluster(self, sample_i, neighbors):
        cluster = [sample_i]
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                # Fetch the samples distant neighbors
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                # Make sure the neighbors neighbors are more than min_samples
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    # Choose neighbors of neighbor except for sample
                    distant_neighbors = self.neighbors[neighbor_i][
                        np.where(self.neighbors[neighbor_i] != sample_i)]
                    # Add the neighbors neighbors as neighbors of sample
                    self.neighbors[sample_i] = np.concatenate(
                        (self.neighbors[sample_i], distant_neighbors))
                    # Expand the cluster from the neighbor
                    expanded_cluster = self._expand_cluster(
                        neighbor_i, self.neighbors[neighbor_i])
                    # Add expanded cluster to this cluster
                    cluster = cluster + expanded_cluster
            if not neighbor_i in np.array(self.clusters):
                cluster.append(neighbor_i)
        return cluster

    # Return the samples labels as the index of the cluster in which they are
    # contained
    def _get_cluster_labels(self):
        # Set default value to number of clusters
        # Will make sure all outliers have same cluster label
        labels = len(self.clusters) * np.ones(np.shape(self.X)[0])
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    # DBSCAN
    def predict(self, X):
        self.X = X
        n_samples = np.shape(self.X)[0]
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = self._get_neighbors(sample_i)
            if len(self.neighbors[sample_i]) >= self.min_samples:
                # If core point => mark as visited
                self.visited_samples.append(sample_i)
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                new_cluster = self._expand_cluster(
                    sample_i, self.neighbors[sample_i])
                # Add cluster to list of clusters
                self.clusters.append(new_cluster)

        # Get the resulting cluster labels
        cluster_labels = self._get_cluster_labels()
        return cluster_labels


def main():
    # Load the dataset
    X, y = datasets.make_moons(n_samples=300, noise=0.1)

    # Cluster the data using DBSCAN
    clf = DBSCAN(eps=0.17, min_samples=5)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred, title="DBSCAN")
    pca.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == "__main__":
    main()
