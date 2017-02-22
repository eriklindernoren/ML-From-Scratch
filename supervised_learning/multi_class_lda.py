import sys, os, scipy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import calculate_covariance_matrix
from data_manipulation import normalize, standardize


class MultiClassLDA():
    def __init__(self, solver="svd"):
        self.solver = solver

    def _calculate_scatter_matrices(self, X, y):
        n_features = np.shape(X)[1]
        labels = np.unique(y)

        # Within class scatter matrix: 
        # SW = sum{ (X_for_class - mean_of_X_for_class)^2 }
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

        # Compute SW^-1 * SB
        A = None
        if self.solver == "svd":
            # Computationally cheaper than other option.
            # Calculate SW^-1 * SB by SVD (pseudoinverse of diagonal matrix S)
            U,S,V = np.linalg.svd(SW)
            S = np.diag(S)
            SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
            A = SW_inverse.dot(SB)
        else:
            # Computationally expensive.
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

    # Plot the dataset X and the corresponding labels y in 2D using the LDA transformation.
    def plot_in_2d(self, X, y):
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:,0]
        x2 = X_transformed[:,1]
        plt.scatter(x1,x2,c=y)
        plt.show()


# Demo
def main():
    # Load the dataset
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    # Project the data onto the 2 primary components
    multi_class_lda = MultiClassLDA()
    multi_class_lda.plot_in_2d(X, y)

if __name__ == "__main__": main()




