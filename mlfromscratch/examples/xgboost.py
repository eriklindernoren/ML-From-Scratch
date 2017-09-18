from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt
import progressbar

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, standardize, to_categorical, normalize
from mlfromscratch.utils.data_operation import mean_squared_error, accuracy_score
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import XGBoost

def main():

    print ("-- XGBoost --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)  

    clf = XGBoost(debug=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, 
        title="XGBoost", 
    accuracy=accuracy, 
    legend_labels=data.target_names)


if __name__ == "__main__":
    main()
