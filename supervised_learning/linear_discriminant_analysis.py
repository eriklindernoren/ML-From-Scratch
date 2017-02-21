import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import calculate_covariance_matrix, calculate_correlation_matrix
from data_manipulation import normalize

# Load dataset and only use the two first classes
data = datasets.load_iris()
X = normalize(data.data[data.target < 2])
y = data.target[data.target < 2]
X1 = X[y == 0]
X2 = X[y == 1]

# Calculate the covariances of the two class distributions
cov1 = calculate_covariance_matrix(X1)
cov2 = calculate_covariance_matrix(X2)
cov_tot = cov1 + cov2

# Get the means of the two class distributions
mean1 = X1.mean(0)
mean2 = X2.mean(0)
mean_diff = np.atleast_1d(mean1 - mean2)

# Calculate w as  (x1_mean - x2_mean) / (cov1 + cov2)
w = np.linalg.inv(cov_tot).dot(mean_diff)

# Project X onto w
x1 = X.dot(w)
x2 = X.dot(w)

# Plot the data
plt.scatter(x1,x2,c=y)
plt.show()