from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import math

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.supervised_learning import RandomForest
from mlfromscratch.utils import Plot

def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)

    clf = RandomForest(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
