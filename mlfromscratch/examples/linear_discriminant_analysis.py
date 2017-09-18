from __future__ import print_function
import sys
import os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
from mlfromscratch.supervised_learning import LDA
from mlfromscratch.utils.data_operation import calculate_covariance_matrix, accuracy_score
from mlfromscratch.utils.data_manipulation import normalize, standardize, train_test_split
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.utils import Plot

def main():
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Three -> two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Fit and predict using LDA
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)

if __name__ == "__main__":
    main()
