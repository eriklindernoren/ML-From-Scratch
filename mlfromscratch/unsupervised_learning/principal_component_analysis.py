from __future__ import print_function
import sys
import os
from sklearn import datasets

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

import numpy as np

from mlfromscratch.utils.data_operation import calculate_covariance_matrix
from mlfromscratch.utils.data_operation import calculate_correlation_matrix
from mlfromscratch.utils.data_manipulation import standardize


class PCA():
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and 
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.
    """
    def __init__(self): pass

    # Fit the dataset to the number of principal components
    # specified in the constructor and return the transformed dataset
    def transform(self, X, n_components):
        covariance = calculate_covariance_matrix(X)

        # Get the eigenvalues and eigenvectors.
        # (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed


def main():

    # Demo of how to reduce the dimensionality of the data to two dimension
    # and plot the results. 

    # Load the dataset
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    X_trans = PCA().transform(X, 2)

    x1 = X_trans[:, 0]
    x2 = X_trans[:, 1]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Add a legend
    plt.legend(class_distr, y, loc=1)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
