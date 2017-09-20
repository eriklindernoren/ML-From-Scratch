from __future__ import division, print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
import pandas as pd

from mlfromscratch.utils.data_manipulation import train_test_split, normalize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils import Plot


class NaiveBayes():
    """The Gaussian Naive Bayes classifier. """
    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        # Gaussian prob. distribution parameters (mean and variance)
        self.parameters = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            x_where_c = X[np.where(y == c)]
            # Add the mean and variance for each feature
            self.parameters.append([])
            for j in range(len(x_where_c[0, :])):
                col = x_where_c[:, j]
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        coeff = (1.0 / (math.sqrt((2.0 * math.pi) * var)))
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return coeff * exponent

    def _calculate_prior(self, c):
        """ Calculate the prior of class c 
        (samples where class == c / total number of samples)"""
        # Selects the rows where the class label is c
        x_where_c = self.X[np.where(self.y == c)]
        n_class_instances = np.shape(x_where_c)[0]
        n_total_instances = np.shape(self.X)[0]
        return n_class_instances / n_total_instances

    def _classify(self, sample):
        """ Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
        P(X|Y) - Likelihood. Gaussian distribution (given by _calculate_likelihood)
        P(Y) - Prior (given by _calculate_prior)
        P(X) - Scales the posterior to make it a proper probability distribution.
               This term is ignored in this implementation since it doesn't affect
               which class distribution the sample is most likely of belonging to.
        Classify the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        for i in range(len(self.classes)):
            c = self.classes[i]
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

