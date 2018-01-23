from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from mlfromscratch.utils import calculate_covariance_matrix, normalize, standardize


class MultiClassLDA():
    """Enables dimensionality reduction for multiple
    class distributions. It transforms the features space into a space where
    the between class scatter is maximized and the within class scatter is
    minimized.

    Parameters:
    -----------
    solver: str
        If 'svd' we use the pseudo-inverse to calculate the inverse of matrices
        when doing the transformation.
    """
    def __init__(self, solver="svd"):
        self.solver = solver

    def _calculate_scatter_matrices(self, X, y):
        n_features = np.shape(X)[1]
        labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum{ (X_for_class - mean_of_X_for_class)^2 }
        #   <=> (n_samples_X_for_class - 1) * covar(X_for_class)
        SW = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]
            SW += (len(_X) - 1) * calculate_covariance_matrix(_X)

        # Between class scatter:
        # SB = sum{ n_samples_for_class * (mean_for_class - total_mean)^2 }
        total_mean = np.mean(X, axis=0)
        SB = np.empty((n_features, n_features))
        for label in labels:
            _X = X[y == label]
            _mean = np.mean(_X, axis=0)
            SB += len(_X) * (_mean - total_mean).dot((_mean - total_mean).T)

        return SW, SB

    def transform(self, X, y, n_components):
        SW, SB = self._calculate_scatter_matrices(X, y)

        # Determine SW^-1 * SB by calculating inverse of SW
        A = np.linalg.inv(SW).dot(SB)

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        # Project the data onto eigenvectors
        X_transformed = X.dot(eigenvectors)

        return X_transformed


    def plot_in_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation."""
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        if title: plt.title(title)
        plt.show()
