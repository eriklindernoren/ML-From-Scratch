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


class DBSCAN():
	def __init__(self, eps=1, min_samples=5):
		self.eps = eps                  	# The radius within which samples are considered neighbors
		self.min_samples = min_samples      # The number of neighbors required for sample to be a core point
		self.clusters = []					# List of arrays (clusters) containing sample indices
		self.visited_samples = []
		self.neighbors = {}					# Hashmap {"sample_index": [neighbor1, neighbor2, ...]}
		self.X = None						# Dataset

	# Return a list of neighboring samples
	# A sample_2 is considered a neighbor of sample_1 if the distance between
	# them is smaller than epsilon
	def _get_neighbors(self, sample_i):
		neighbors = []
		for _sample_i, _sample in enumerate(self.X):
			if _sample_i != sample_i and euclidean_distance(self.X[sample_i], _sample) < self.eps:
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
					distant_neighbors = self.neighbors[neighbor_i][np.where(self.neighbors[neighbor_i] != sample_i)]
					# Add the neighbors neighbors as neighbors of sample
					self.neighbors[sample_i] = np.concatenate((self.neighbors[sample_i], distant_neighbors))
					# Expand the cluster from the neighbor
					expanded_cluster = self._expand_cluster(neighbor_i, self.neighbors[neighbor_i])
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
			self.visited_samples.append(sample_i)
			self.neighbors[sample_i] = self._get_neighbors(sample_i)
			if len(self.neighbors[sample_i]) >= self.min_samples:
				# Sample has more neighbors than self.min_samples => expand cluster from sample
				new_cluster = self._expand_cluster(sample_i, self.neighbors[sample_i])
				# Add cluster to list of clusters
				self.clusters.append(new_cluster)

		# Get the resulting cluster labels
		cluster_labels = self._get_cluster_labels()
		return cluster_labels


# Demo
def main():
    # Load the dataset
    X, y = datasets.make_moons(noise=0.05)

    # Cluster the data using DBSCAN
    clf = DBSCAN(eps=0.3, min_samples=5)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred)
    pca.plot_in_2d(X, y)

if __name__ == "__main__": main()




