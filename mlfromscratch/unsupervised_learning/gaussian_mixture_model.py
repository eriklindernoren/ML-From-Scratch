from __future__ import division, print_function
import sys
import os
import math
import random
from sklearn import datasets
import numpy as np

# Import helper functions
from mlfromscratch.utils.data_manipulation import normalize
from mlfromscratch.utils.data_operation import euclidean_distance, calculate_covariance_matrix
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils import Plot


class GaussianMixtureModel():
    """A probabilistic clustering method for determining groupings among data samples.

    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that. 
    tolerance: float
        If the difference of the results from one iteration to the next is
        smaller than this value we will say that the algorithm has converged.
    """
    def __init__(self, k=2, max_iterations=2000, tolerance=1e-8):
        self.k = k
        self.parameters = []
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.responsibilities = []
        self.sample_assignments = None
        self.responsibility = None

    # Initialize gaussian randomly
    def _init_random_gaussians(self, X):
        n_samples = np.shape(X)[0]
        self.priors = (1 / self.k) * np.ones(self.k)
        for i in range(self.k):
            params = {}
            params["mean"] = X[np.random.choice(range(n_samples))]
            params["cov"] = calculate_covariance_matrix(X)
            self.parameters.append(params)

    # Likelihood
    def multivariate_gaussian(self, X, params):
        n_features = np.shape(X)[1]
        mean = params["mean"]
        covar = params["cov"]
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            d = n_features  # dimension
            coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2)
                            * math.sqrt(determinant)))
            exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

    # Calculate the likelihood over all samples
    def _get_likelihoods(self, X):
        n_samples = np.shape(X)[0]
        likelihoods = np.zeros((n_samples, self.k))
        for i in range(self.k):
            likelihoods[
                :, i] = self.multivariate_gaussian(
                X, self.parameters[i])
        return likelihoods

    # Calculate the responsibility
    def _expectation(self, X):
        # Calculate probabilities of X belonging to the different clusters
        weighted_likelihoods = self._get_likelihoods(X) * self.priors
        sum_likelihoods = np.expand_dims(
            np.sum(weighted_likelihoods, axis=1), axis=1)
        # Determine responsibility as P(X|y)*P(y)/P(X)
        self.responsibility = weighted_likelihoods / sum_likelihoods
        # Assign samples to cluster that has largest probability
        self.sample_assignments = self.responsibility.argmax(axis=1)
        # Save value for convergence check
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    # Update the parameters and priors
    def _maximization(self, X):
        # Iterate through clusters and recalculate mean and covariance
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = (resp * X).sum(axis=0) / resp.sum()
            covariance = (X - mean).T.dot((X - mean) * resp) / resp.sum()
            self.parameters[i]["mean"], self.parameters[
                i]["cov"] = mean, covariance

        # Update weights
        n_samples = np.shape(X)[0]
        self.priors = self.responsibility.sum(axis=0) / n_samples

    # Covergence if || likehood - last_likelihood || < tolerance
    def _converged(self, X):
        if len(self.responsibilities) < 2:
            return False
        diff = np.linalg.norm(
            self.responsibilities[-1] - self.responsibilities[-2])
        # print ("Likelihood update: %s (tol: %s)" % (diff, self.tolerance))
        return diff <= self.tolerance

    # Run GMM and return the cluster indices
    def predict(self, X):
        # Initialize the gaussians randomly
        self._init_random_gaussians(X)

        # Run EM until convergence or for max iterations
        for _ in range(self.max_iterations):
            self._expectation(X)    # E-step
            self._maximization(X)   # M-step

            # Check convergence
            if self._converged(X):
                break

        # Make new assignments and return them
        self._expectation(X)
        return self.sample_assignments

def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data
    clf = GaussianMixtureModel(k=3)
    y_pred = clf.predict(X)

    p = MatplotlibWrapper()
    p.plot_in_2d(X, y_pred, title="GMM Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()
