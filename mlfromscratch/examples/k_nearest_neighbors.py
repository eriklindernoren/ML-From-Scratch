from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from mlfromscratch.utils import train_test_split, normalize, accuracy_score
from mlfromscratch.utils import euclidean_distance, Plot
from mlfromscratch.supervised_learning import KNN

def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()