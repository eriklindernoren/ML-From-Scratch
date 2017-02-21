import sys, os, scipy
from sklearn import datasets
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

    # Calculate the mean vectors for each class
    def _calculate_mean_vectors(self, X, y):
        mean_vectors = []
        for label in np.unique(y):
            _X = X[y == label]
            mean_vector = np.mean(_X, axis=0)
            mean_vectors.append(mean_vector)
        return np.array(mean_vectors)

    def _calculate_scatter_matrices(self, X, y, mean_vectors):
        # Within class scatter matrix
        n_features = np.shape(X)[1]
        SW = np.empty((n_features, n_features))
        labels = np.unique(y)
        samples_per_dist = []
        for label in labels:
            _X = X[y == label]
            n_samples = np.shape(_X)[0]
            samples_per_dist.append(n_samples)
            within_scatter = None
            if self.solver == "svd":
                U,S,V = np.linalg.svd((_X - np.mean(_X, axis=0)))
                S = np.diag(S)
                # (_X - mean).T.dot(_X - mean) = V.T.dot(S.T).dot(S).dot(V) (By SVD)
                within_scatter = V.dot(S.T).dot(S).dot(V.T)
            else:
                within_scatter = (n_samples - 1) * calculate_covariance_matrix(_X)
            SW += within_scatter

        # Between class scatter matrix
        total_mean = np.mean(X, axis=0)
        SB = np.empty(np.shape(SW))
        for i, mean_vector in enumerate(mean_vectors):
            n_samples = samples_per_dist[i]
            between_scatter = None
            if self.solver == "svd":
                U,S,V = np.linalg.svd(np.expand_dims(mean_vector - total_mean, axis=1))
                S = np.diag(S)
                between_scatter = n_samples * V.dot(S.T).dot(S).dot(V.T)
            else:
                between_scatter = n_samples * (mean_vector - total_mean).dot((mean_vector - total_mean).T)
            SB += between_scatter

        return SW, SB
        

    def transform(self, X, y, n_components):

        mean_vectors = self._calculate_mean_vectors(X, y)
        SW, SB = self._calculate_scatter_matrices(X, y, mean_vectors)

        # Solve for eigenvalues and eigenvectors 
        A = None
        if self.solver == "svd":
            U,S,V = np.linalg.svd(SW)
            S = np.diag(S)
            SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
            A = SW_inverse.dot(SB)
        else:
            A = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        # Project the data onto principal components
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

    # Project the data onto the 2 primary principal components and plot the data
    multi_class_lda = MultiClassLDA()
    multi_class_lda.plot_in_2d(X, y)

if __name__ == "__main__": main()




