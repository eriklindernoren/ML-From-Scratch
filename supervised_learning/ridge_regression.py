import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sys, os
# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import k_fold_cross_validation_sets
from data_operation import mean_squared_error

class RidgeRegression():
	def __init__(self, delta):
		self.w = None
		self.delta = delta # Regularization constant

	def fit(self, X, y):
		# Insert constant ones for bias weights
		X = np.insert(X, 0, 1, axis=1)
		n_features = np.shape(X)[1]

		# Get weights by least squares with regularization (by pseudoinverse)
		U,S,V = np.linalg.svd(X.T.dot(X) + self.delta * np.identity(n_features))
		S = np.diag(S)
		X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
		self.w = X_sq_reg_inv.dot(X.T).dot(y)

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
	X = np.expand_dims(diabetes.data[:, 2], 1)

	# Split the data into training/testing sets
	X_train, X_test = np.array(X[:-20]), np.array(X[-20:])

	# Split the targets into training/testing sets
	y_train, y_test = np.array(diabetes.target[:-20]), np.array(diabetes.target[-20:])

	# Finding regularization constant using cross validation
	lowest_error = float("inf")
	best_reg_factor = None
	print "Finding regularization constant using cross validation:"
	k = 10
	for regularization_factor in np.arange(0,0.5,0.0001):
		cross_validation_sets = k_fold_cross_validation_sets(X_train, y_train, k=k)
		mse = 0
		for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
			clf = RidgeRegression(delta=regularization_factor)
			clf.fit(_X_train, _y_train)
			y_pred = clf.predict(_X_test)
			_mse = mean_squared_error(_y_test, y_pred)
			mse += _mse
		mse /= k

		# Print the mean squared error
		print "\tMean Squared Error: %s (regularization: %s)" % (mse, regularization_factor)

		# Save reg. constant that gave lowest error
		if mse < lowest_error:
			best_reg_factor = regularization_factor
			lowest_error = mse

	# Make final prediction
	clf = RidgeRegression(delta=best_reg_factor)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print "Mean squared error: %s (given by reg. factor: %s)" % (lowest_error, best_reg_factor)
	# Plot the results
	plt.scatter(X_test[:,0], y_test,  color='black')
	plt.plot(X_test[:,0], y_pred, color='blue', linewidth=3)
	plt.show()

if __name__ == "__main__": main()


