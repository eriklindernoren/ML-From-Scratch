from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix

# Dataset
data = datasets.load_iris()
X = data.data
y = data.target

n_features = len(X[0])

# Calculate the covariance matrix for the data
covariance = calculate_covariance_matrix(X)

# Get the eigenvalues and eigenvectors. (eigenvector[:,0] corresponds to eigenvalue[0])
eigenvalues,eigenvectors = np.linalg.eig(covariance)

# Sort the eigenvalues and corresponding eigenvectors from largest
# to smallest eigenvalue. 
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Get two first principal components
evect1 = eigenvectors[:,0].reshape((n_features,1))
evect2 = eigenvectors[:,1].reshape((n_features,1))

# Project data onto the first two principal components
a = X.dot(evect1)
b = X.dot(evect2)

# Plot the data
plt.scatter(a,b,c=y)
plt.show()
