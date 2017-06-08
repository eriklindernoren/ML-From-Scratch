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


class PAM():
    """A simple clustering method that forms k clusters by first assigning
    samples to the closest medoids, and then swapping medoids with non-medoid
    samples if the total distance (cost) between the cluster members and their medoid
    is smaller than prevoisly.


    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    """
    def __init__(self, k=2):
        self.k = k

    # Initialize the medoids as random samples
    def _init_random_medoids(self, X):
        n_samples, n_features = np.shape(X)
        medoids = np.zeros((self.k, n_features))
        for i in range(self.k):
            medoid = X[np.random.choice(range(n_samples))]
            medoids[i] = medoid
        return medoids

    # Return the index of the closest medoid to the sample
    def _closest_medoid(self, sample, medoids):
        closest_i = None
        closest_distance = float("inf")
        for i, medoid in enumerate(medoids):
            distance = euclidean_distance(sample, medoid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    # Assign the samples to the closest medoids to create clusters
    def _create_clusters(self, X, medoids):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            medoid_i = self._closest_medoid(sample, medoids)
            clusters[medoid_i].append(sample_i)
        return clusters

    # Calculate the cost (total distance between samples and their medoids)
    def _calculate_cost(self, X, clusters, medoids):
        cost = 0
        # For each cluster
        for i, cluster in enumerate(clusters):
            medoid = medoids[i]
            for sample_i in cluster:
                # Add distance between sample and medoid as cost
                cost += euclidean_distance(X[sample_i], medoid)
        return cost

    # Returns a list of all samples that are not currently medoids
    def _get_non_medoids(self, X, medoids):
        non_medoids = []
        for sample in X:
            if not sample in medoids:
                non_medoids.append(sample)
        return non_medoids

    # Classify samples as the index of their clusters
    def _get_cluster_labels(self, clusters, X):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i in range(len(clusters)):
            cluster = clusters[cluster_i]
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Do Partitioning Around Medoids and return the cluster labels
    def predict(self, X):
        # Initialize medoids randomly
        medoids = self._init_random_medoids(X)
        # Assign samples to closest medoids
        clusters = self._create_clusters(X, medoids)

        # Calculate the initial cost (total distance between samples and
        # corresponding medoids)
        cost = self._calculate_cost(X, clusters, medoids)

        # Iterate until we no longer have a cheaper cost
        while True:
            best_medoids = medoids
            lowest_cost = cost
            for medoid in medoids:
                # Get all non-medoid samples
                non_medoids = self._get_non_medoids(X, medoids)
                # Calculate the cost when swapping medoid and samples
                for sample in non_medoids:
                    # Swap sample with the medoid
                    new_medoids = medoids.copy()
                    new_medoids[medoids == medoid] = sample
                    # Assign samples to new medoids
                    new_clusters = self._create_clusters(X, new_medoids)
                    # Calculate the cost with the new set of medoids
                    new_cost = self._calculate_cost(
                        X, new_clusters, new_medoids)
                    # If the swap gives us a lower cost we save the medoids and cost
                    if new_cost < lowest_cost:
                        lowest_cost = new_cost
                        best_medoids = new_medoids
            # If there was a swap that resultet in a lower cost we save the
            # resulting medoids from the best swap and the new cost 
            if lowest_cost < cost:
                cost = lowest_cost
                medoids = best_medoids 
            # Else finished
            else:
                break

        final_clusters = self._create_clusters(X, medoids)
        # Return the samples cluster indices as labels
        return self._get_cluster_labels(final_clusters, X)


def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Medoids
    clf = PAM(k=3)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred, title="PAM Clustering")
    pca.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()
