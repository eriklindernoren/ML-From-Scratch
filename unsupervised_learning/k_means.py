import sys, os, math, random
from sklearn import datasets
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import euclidean_distance, normalize
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class KMeans():
	def __init__(self, k=2, max_iterations=500):
		self.k = k
		self.max_iterations = max_iterations

	# Initialize the centroids as random samples
	def _init_random_centroids(self, X):
		n_samples = np.shape(X)[0]
		n_features = np.shape(X)[1]
		centroids = np.zeros((self.k, n_features))
		for i in range(self.k):
			centroid = X[np.random.choice(range(n_samples))]
			centroids[i] = centroid
		return centroids

	# Return the index of the closest centroid to the sample
	def _closest_centroid(self, sample, centroids):
		closest_i = None
		closest_distance = float("inf")
		for i, centroid in enumerate(centroids):
			distance = euclidean_distance(sample, centroid)
			if distance < closest_distance:
				closest_i = i
				closest_distance = distance
		return closest_i

	# Assign the samples to the closest centroids to create clusters
	def _create_clusters(self, centroids, X):
		n_samples = np.shape(X)[0]
		clusters = [[] for _ in range(self.k)]
		for sample_i, sample in enumerate(X):
			centroid_i = self._closest_centroid(sample, centroids)
			clusters[centroid_i].append(sample_i)
		return clusters

	# Calculate new centroids as the means of the samples
	# in each cluster
	def _get_centroids(self, clusters, X):
		n_features = np.shape(X)[1]
		centroids = np.zeros((self.k, n_features))
		for i, cluster in enumerate(clusters):
			centroid = np.mean(X[cluster], axis=0)
			centroids[i] = centroid
		return centroids

	# Classify samples as the index of their clusters
	def _classify(self, clusters, X):
		y_pred = np.zeros(np.shape(X)[0])
		for cluster_i in range(len(clusters)):
			cluster = clusters[cluster_i]
			for sample_i in cluster:
				y_pred[sample_i] = cluster_i
		return y_pred

	def predict(self, X):
		# Initialize centroids
		centroids = self._init_random_centroids(X)

		# Iterate until convergence or for max iterations
		for _ in range(self.max_iterations):
			# Assign samples to closest centroids (create clusters)
			clusters = self._create_clusters(centroids, X)
			prev_centroids = centroids
			# Calculate new centroids from the clusters
			centroids = self._get_centroids(clusters, X)

			diff = centroids - prev_centroids
			# If not any centroid have changed => convergence
			if not diff.any():
				break

		return self._classify(clusters, X)



# Demo
def main():
    # Load the dataset
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    clf = KMeans(k=10)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred)
    pca.plot_in_2d(X, y)



if __name__ == "__main__": main()


