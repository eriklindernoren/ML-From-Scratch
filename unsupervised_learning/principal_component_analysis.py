import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    # Get the covariance of the dataset X
    def get_covariance(self, X):
        # Calculate the covariance matrix for the data
        covariance = calculate_covariance_matrix(X,X)
        return covariance

    # Fit the dataset to the number of principal components specified in the constructor
    # and return the transform dataset
    def fit(self, X):
        covariance = self.get_covariance(X)

        # Get the eigenvalues and eigenvectors. (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue. 
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        # Get two first principal components
        evects = np.atleast_1d(eigenvectors[:,0:self.n_components])

        # Project the data onto principal components
        X_transformed = X.dot(evects)

        return X_transformed

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None):
        X_transformed = self.fit(X)
        x1 = X_transformed[:,0]
        x2 = X_transformed[:,1]
        plt.scatter(x1,x2,c=y)
        plt.show()


# Demo of the pca module
def main():
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(n_components=2)

    # Plot the data in 2d
    pca.plot_in_2d(X, y)

if __name__ == "__main__": main()




