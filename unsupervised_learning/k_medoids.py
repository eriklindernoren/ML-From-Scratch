import sys, os, math, random
from sklearn import datasets
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import normalize
from data_operation import euclidean_distance
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class KMedoids():
	def __init__(self, k=2, max_iterations=500):
		self.k = k
		self.max_iterations = max_iterations

	# Initialize the medoids as random samples
	def _init_random_medoids(self, X):
		n_samples = np.shape(X)[0]
		n_features = np.shape(X)[1]
		medoids = np.zeros((self.k, n_features))
		for i in range(self.k):
			centroid = X[np.random.choice(range(n_samples))]
			medoids[i] = centroid
		return medoids

	# Return the index of the closest centroid to the sample
	def _closest_centroid(self, sample, medoids):
		closest_i = None
		closest_distance = float("inf")
		for i, centroid in enumerate(medoids):
			distance = euclidean_distance(sample, centroid)
			if distance < closest_distance:
				closest_i = i
				closest_distance = distance
		return closest_i

	# Assign the samples to the closest medoids to create clusters
	def _create_clusters(self, medoids, X):
		n_samples = np.shape(X)[0]
		clusters = [[] for _ in range(self.k)]
		for sample_i, sample in enumerate(X):
			centroid_i = self._closest_centroid(sample, medoids)
			clusters[centroid_i].append(sample_i)
		return clusters

	# Calculate new medoids as the median of the samples
	# in each cluster
	def _calculate_medoids(self, clusters, X):
		n_features = np.shape(X)[1]
		medoids = np.zeros((self.k, n_features))
		for i, cluster in enumerate(clusters):
			centroid = np.median(X[cluster], axis=0)
			medoids[i] = centroid
		return medoids

	# Classify samples as the index of their clusters
	def _get_cluster_labels(self, clusters, X):
		# One prediction for each sample
		y_pred = np.zeros(np.shape(X)[0])
		for cluster_i in range(len(clusters)):
			cluster = clusters[cluster_i]
			for sample_i in cluster:
				y_pred[sample_i] = cluster_i
		return y_pred

	# Do K-Means clustering and return cluster indices
	def predict(self, X):
		# Initialize medoids
		medoids = self._init_random_medoids(X)

		# Iterate until convergence or for max iterations
		for _ in range(self.max_iterations):
			# Assign samples to closest medoids (create clusters)
			clusters = self._create_clusters(medoids, X)
			prev_medoids = medoids
			# Calculate new medoids from the clusters
			medoids = self._calculate_medoids(clusters, X)

			# If no medoids have changed => convergence
			diff = medoids - prev_medoids
			if not diff.any():
				break

		return self._get_cluster_labels(clusters, X)


# Demo
def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Medoids
    clf = KMedoids(k=3)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred)
    pca.plot_in_2d(X, y)



if __name__ == "__main__": main()


