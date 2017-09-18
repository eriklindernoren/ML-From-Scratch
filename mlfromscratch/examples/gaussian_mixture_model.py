from __future__ import division, print_function
import sys
import os
import math
import random
from sklearn import datasets
import numpy as np

from mlfromscratch.unsupervised_learning import GaussianMixtureModel
from mlfromscratch.utils import Plot


def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data
    clf = GaussianMixtureModel(k=3)
    y_pred = clf.predict(X)

    p = Plot()
    p.plot_in_2d(X, y_pred, title="GMM Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()