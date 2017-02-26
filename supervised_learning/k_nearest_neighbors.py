import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, normalize
from data_operation import euclidean_distance, accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

class KNN():
	def __init__(self, k=5):
		self.k = k

	# Do a majority vote among the neighbors
	def _get_vote(self, neighbors, classes):
		max_count = 0
		label = None
		# Count class occurences among neighbors
		for c in classes:
			count = 0
			for sample in neighbors:
				sample_class = sample[1]
				if sample_class == c:
					count += 1
			# If vote is larger than highest previous => update label pred.
			if count > max_count:
				max_count = count
				label = c
		return label

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
			k_nearest_neighbors = neighbors[neighbors[:,0].argsort()][:self.k]
			# Do a majority vote among the k neighbors and set prediction as the 
			# class receing the most votes
			label = self._get_vote(k_nearest_neighbors, classes)
			y_pred.append(label)
		return np.array(y_pred)

# Demo
def main():
	iris = load_iris()
	X = normalize(iris.data)
	y = iris.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	clf = KNN(k=3)
	y_pred = clf.predict(X_test, X_train, y_train)
	print "Accuracy score:", accuracy_score(y_test, y_pred)

	# Reduce dimensions to 2d using pca and plot the results
	pca = PCA()
	pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__": main()



