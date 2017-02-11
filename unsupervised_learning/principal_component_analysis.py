import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix

# Dataset
data = datasets.load_iris()
X = data.data
y = data.target

n_features = len(X[0])

# Calculate the covariance matrix for the data
covariance = calculate_covariance_matrix(X,X)

# Get the eigenvalues and eigenvectors. (eigenvector[:,0] corresponds to eigenvalue[0])
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# Sort the eigenvalues and corresponding eigenvectors from largest
# to smallest eigenvalue. 
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Get two first principal components
evect1 = np.atleast_1d(eigenvectors[:,0])
evect2 = np.atleast_1d(eigenvectors[:,1])

# Project data onto the first two principal components
a = X.dot(evect1)
b = X.dot(evect2)

# Plot the data
plt.scatter(a,b,c=y)
plt.show()
