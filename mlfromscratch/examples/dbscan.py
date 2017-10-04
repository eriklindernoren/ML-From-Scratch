import sys
import os
import math
import random
from sklearn import datasets
import numpy as np

# Import helper functions
from mlfromscratch.utils import Plot
from mlfromscratch.unsupervised_learning import DBSCAN

def main():
    # Load the dataset
    X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

    # Cluster the data using DBSCAN
    clf = DBSCAN(eps=0.17, min_samples=5)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="DBSCAN")
    p.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == "__main__":
    main()
