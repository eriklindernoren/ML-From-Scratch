import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sys, os
# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import mean_squared_error

class LinearRegression():
	def __init__(self):
		self.w = None

	def fit(self, X, y):
		# Insert constant ones for bias weights
		X = np.insert(X, 0, 1, axis=1)	
		# Get weights by least squares (by pseudoinverse)
		U,S,V = np.linalg.svd(X.T.dot(X))
		S = np.diag(S)
		X_sq_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
		self.w = X_sq_inv.dot(X.T).dot(y)

	def predict(self, X):
		# Insert constant ones for bias weights
		X = np.insert(X, 0, 1, axis=1)
		y_pred = X.dot(self.w)
		return y_pred

# Demo
def main():
	# Load the diabetes dataset
	diabetes = datasets.load_diabetes()

	# Use only one feature
	X = diabetes.data[:, np.newaxis, 2]

	# Split the data into training/testing sets
	x_train, x_test = X[:-20], X[-20:]

	# Split the targets into training/testing sets
	y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]

	clf = LinearRegression()
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	# Print the mean squared error
	print "Mean Squared Error:", mean_squared_error(y_test, y_pred)

	# Plot the results
	plt.scatter(x_test[:,0], y_test,  color='black')
	plt.plot(x_test[:,0], y_pred, color='blue', linewidth=3)
	plt.show()

if __name__ == "__main__": main()


