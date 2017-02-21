import sys, os
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import calculate_covariance_matrix, calculate_correlation_matrix
from data_manipulation import standardize


class PCA():
    def __init__(self): pass

    # Get the covariance of X
    def get_covariance(self, X):
        # Calculate the covariance matrix for the data
        covariance = calculate_covariance_matrix(X)
        return covariance

    # Fit the dataset to the number of principal components specified in the constructor
    # and return the transform dataset
    def transform(self, X, n_components):
        covariance = self.get_covariance(X)

        # Get the eigenvalues and eigenvectors. (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:,idx])[:,:n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None):
        X_transformed = self.transform(X, n_components=2)
        x1 = X_transformed[:,0]
        x2 = X_transformed[:,1]
        plt.scatter(x1,x2,c=y)
        plt.show()

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_3d(self, X, y=None):
        X_transformed = self.transform(X, n_components=3)
        x1 = X_transformed[:,0]
        x2 = X_transformed[:,1]
        x3 = X_transformed[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1,x2,x3,c=y)
        plt.show()

# Demo
def main():
    # Load the dataset
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components and plot the data
    pca = PCA()
    pca.plot_in_2d(X, y)

if __name__ == "__main__": main()




