from __future__ import print_function
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.utils import make_diagonal, normalize, train_test_split, accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import LogisticRegression

def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)

if __name__ == "__main__":
    main()