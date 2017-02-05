
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

k = 5
classes = np.unique(y_train)

def euclidean_distance(instance1, instance2):
	distance = 0
	for x in range(len(instance1)):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def get_vote(neighbors):
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

def predict(X):
	y_pred = []
	for i in range(len(X)):
		test_sample = X[i]
		neighbors = []
		for j in range(len(X_train)):
			observed_sample = X_train[j]
			distance = euclidean_distance(test_sample, observed_sample)
			label = y_train[j]
			neighbors.append([distance, label])
		neighbors = np.array(neighbors)
		k_closest_neighbors = neighbors[neighbors[:,0].argsort()][:k]
		label = get_vote(k_closest_neighbors)
		y_pred.append(label)
	return np.array(y_pred)

y_pred = predict(X_test)
print "Accuracy score:", accuracy_score(y_test, y_pred)


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the data
plt.scatter(X[:][:, 0], X[:][:,1],c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

