import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import calculate_covariance_matrix
from data_manipulation import normalize, standardize


class LDA():
	def __init__(self): pass

	def transform(self, X, y):
		# Separate data by class
		X1 = X[y == 0]
		X2 = X[y == 1]

		# Calculate the covariance matrices of the two datasets
		cov1 = calculate_covariance_matrix(X1)
		cov2 = calculate_covariance_matrix(X2)
		cov_tot = cov1 + cov2

		# Calculate the mean of the two datasets
		mean1 = X1.mean(0)
		mean2 = X2.mean(0)
		mean_diff = np.atleast_1d(mean1 - mean2)

		# Determine the vector which when X is projected onto it best separates the
		# data by class. w = (mean1 - mean2) / (cov1 + cov2)
		w = np.linalg.pinv(cov_tot).dot(mean_diff)

		# Project data onto vector
		X_transform = X.dot(w)

		return X_transform

# Demo
def main():
	# Load the dataset
	data = datasets.load_iris()
	X = data.data
	y = data.target

	# Three -> two classes
	X = X[y != 2]
	y = y[y != 2]

	# Transform data using LDA
	lda = LDA()
	X_transformed = lda.transform(X, y)

	# Plot the data
	plt.scatter(X_transformed, X_transformed,c=y)
	plt.show()

if __name__ == "__main__": main()
