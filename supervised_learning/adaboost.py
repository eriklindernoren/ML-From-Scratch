from __future__ import division
import math, sys, os
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split
from data_operation import accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        # List of weak classifiers
        # clf = [threshold, polarity, feature_index, alpha]
        self.clfs = np.ones((4,self.n_clf))

    def fit(self, X, y):
        
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.ones(n_samples)*(1/n_features)
        # Iterate through classifiers
        for c in range(self.n_clf):
            # Initial values
            err_min = 1
            polarity = 0
            threshold = 0
            feature_index = 0
            # Iterate throught every sample for each feature
            for n in range(n_features):
                for t in range(n_samples):
                    err = 0
                    # Select feature as threshold
                    tao = X[t, n]
                    # Iterate through all values of the threshold feature and measure the value
                    # against the threshold and determine by the error if makes for a good predictor
                    for m in range(n_samples):
                        x = X[m, n]
                        y_true = y[m]
                        h = 1.0             # Hypothesis
                        p = 1               # Polarity
                        if p*x < p*tao:     # If lower than threshold => h = -1
                            h = -1.0
                        err = err + w[m]*(y_true != h)
                    # E.g err = 0.8 => (1 - err) = 0.2
                    # We flip the error and polarity
                    if err > 0.5 and err < 1:
                        err = 1 - err
                        p = -1
                    # If this threshold resulted in the smallest error we save the
                    # the configuration
                    if err < err_min:
                        polarity = p
                        threshold = tao
                        feature_index = n
                        err_min = err
            # Calculate the alpha which is used to update the sample weights
            # and is an approximation of this classifiers proficiency
            alpha = 0.5*math.log((1.0001-err_min)/(err_min + 0.0001))
            # Save the classifier configuration
            self.clfs[:4,c] = [threshold, polarity, feature_index, alpha]
            # Iterate through samples and update weights 
            # Large weight => hard sample to classify
            for m in range(n_samples):
                h = 1.0
                x = X[m, feature_index]
                y_true = y[m]
                if polarity*x < polarity*threshold:
                    h = -1.0
                w[m] = w[m]*math.exp(-alpha*y_true*h)
            # Renormalize the weight vector
            w = w * (1/np.sum(w))

    def predict(self, X):
        y_pred = []
        correct = 0
        # Iterate through each test sample and classify by commitee
        for i in range(len(X[:,0])):
            s = 0
            for c in range(self.n_clf):
                # Get classifier configuration
                clf = self.clfs[:, c]
                threshold = clf[0]
                polarity = clf[1]
                feature_index = int(clf[2])
                alpha = clf[3]
                x = X[i, feature_index]
                h = 1
                if polarity*x < polarity*threshold:
                    h = -1
                # Weight prediction by classifiers approximated proficiency
                s += alpha*h
            y = np.sign(s)
            y_pred.append(y)
        return y_pred

# Demo
def main():
    df = pd.read_csv(dir_path + "/../data/iris.csv")
    # Change class labels from strings to numbers
    df = df.replace(to_replace="setosa", value="-1")
    df = df.replace(to_replace="virginica", value="1")
    df = df.replace(to_replace="versicolor", value="2")

    # Only select data for two classes
    X = df.loc[df['species'] != "2"].drop("species", axis=1).as_matrix()
    y = df.loc[df['species'] != "2"]["species"].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    # Adaboost classification
    clf = Adaboost(n_clf = 8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

    # Reduce dimensions to 2d using pca and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__": main()
