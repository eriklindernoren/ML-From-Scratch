from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import sys
import os
import math
# Import helper functions
from mlfromscratch.utils.data_manipulation import divide_on_feature, train_test_split, get_random_subsets, normalize
from mlfromscratch.utils.data_operation import accuracy_score, calculate_entropy
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.supervised_learning import ClassificationTree


class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further. 
    max_depth: int
        The maximum depth of a tree.
    debug: boolean
        True or false depending on if we wish to display the training errors.
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=1e-7, max_depth=float("inf"), debug=False):
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree
        self.feature_indices = []           # The indices of the features used for each tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain req. to continue
        self.max_depth = max_depth          # Maximum depth for tree
        self.debug = debug

        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        if self.debug:
            print ("Training (%s estimators):" % (self.n_estimators))
        # Choose one random subset of the data for each tree
        subsets = get_random_subsets(X, y, self.n_estimators)
        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.feature_indices.append(idx)
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)

            if self.debug:
                progress = 100 * (i / self.n_estimators)
                print ("Progress: %.2f%%" % progress)

    def predict(self, X):
        y_preds = []
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Select the features that the tree has trained on
            idx = self.feature_indices[i]
            # Make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds.append(prediction)
            
        # Take the transpose of the matrix to transform it so
        # that rows are samples and columns are predictions by the
        # estimators
        y_preds = np.array(y_preds).T
        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Do a majority vote over the predictions (columns)
            max_count = 0
            most_common = None
            # For each unique predicted label -> count occurences
            # and save the most predicted label
            for label in np.unique(sample_predictions):
                count = len(sample_predictions[sample_predictions == label])
                if count > max_count:
                    max_count = count
                    most_common = label
            # The most common prediction gets added as final prediction 
            # of the sample
            y_pred.append(most_common)
        return y_pred


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

    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
