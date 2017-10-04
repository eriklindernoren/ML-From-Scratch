from __future__ import division, print_function
import numpy as np
import math
from mlfromscratch.utils import train_test_split, normalize
from mlfromscratch.utils import Plot, accuracy_score


class NaiveBayes():
    """The Gaussian Naive Bayes classifier. """
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # Add the mean and variance for each feature (column)
            for j in range(X.shape[1]):
                col = X_where_c[:, j]
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        """ Calculate the prior of class c 
        (samples where class == c / total number of samples)"""
        X_where_c = self.X[np.where(self.y == c)]
        n_class_instances = X_where_c.shape[0]
        n_total_instances = self.X.shape[0]
        return n_class_instances / n_total_instances

    def _classify(self, sample):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X)

        P(X|Y) - Likelihood of data X given class distribution Y. 
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.

        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Multiply with the class likelihoods
            for j, params in enumerate(self.parameters[i]):
                sample_feature = sample[j]
                # Determine P(x|Y)
                likelihood = self._calculate_likelihood(params["mean"], params["var"], sample_feature)
                # Multiply with the accumulated probability
                posterior *= likelihood
            # Total posterior = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            posteriors.append(posterior)
        # Return the class with the largest posterior probability
        index_of_max = np.argmax(posteriors)
        return self.classes[index_of_max]

    def predict(self, X):
        """ Predict the class labels of the samples in X """
        y_pred = []
        for sample in X:
            y = self._classify(sample)
            y_pred.append(y)
        return y_pred
