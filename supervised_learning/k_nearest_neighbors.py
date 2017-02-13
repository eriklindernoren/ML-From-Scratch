import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import train_test_split, accuracy_score, euclidean_distance
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

class KNN():
	def __init__(self, k=5):
		self.k = k

	def _get_vote(self, neighbors, classes):
		max_count = 0
		label = None
		for c in classes:
			count = 0
			for neighbor in neighbors:
				if neighbor[1] == c:
					count += 1
			if count > max_count:
				max_count = count
				label = c
		return label

	def predict(self, x_test, x_train, y_train):
		classes = np.unique(y_train)
		y_pred = []
		for i in range(len(x_test)):
			test_sample = x_test[i]
			neighbors = []
			for j in range(len(x_train)):
				observed_sample = x_train[j]
				distance = euclidean_distance(test_sample, observed_sample)
				label = y_train[j]
				neighbors.append([distance, label])
			neighbors = np.array(neighbors)
			k_nearest_neighbors = neighbors[neighbors[:,0].argsort()][:self.k]
			label = self._get_vote(k_nearest_neighbors, classes)
			y_pred.append(label)
		return np.array(y_pred)

# Demo of the knn classifier
def main():
	iris = load_iris()
	X = iris.data
	y = iris.target
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	clf = KNN(k=3)
	y_pred = clf.predict(x_test, x_train, y_train)
	print "Accuracy score:", accuracy_score(y_test, y_pred)

	# Reduce dimensions to 2d using pca and plot the results
	pca = PCA(n_components=2)
	X_transformed = pca.transform(x_test)
	x1 = X_transformed[:,0]
	x2 = X_transformed[:,1]

	plt.scatter(x1,x2,c=y_pred)
	plt.show()


if __name__ == "__main__": main()



