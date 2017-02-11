from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix

# Dataset
data = datasets.load_iris()
X = data.data
y = data.target

# Calculate the covariance matrix for the data
covariance = calculate_covariance_matrix(X)

# Get the eigenvalues and eigenvectors. (eigenvector[:,0] corresponds to eigenvalue[0])
eigenvalues,eigenvectors = np.linalg.eig(covariance)

# Sort the eigenvalues and corresponding eigenvectors from largest
# to smallest eigenvalue. 
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]


# Project data onto the first two principal components
a = X.dot(eigenvectors[:,0].reshape((len(eigenvectors[0]),1)))
b = X.dot(eigenvectors[:,1].reshape((len(eigenvectors[1]),1)))

# Plot the data
plt.scatter(a,b,c=y)
plt.show()
