from __future__ import division
import numpy as np
from sklearn import datasets
import sys
import os
import matplotlib.pyplot as plt

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import divide_on_feature, train_test_split, standardize
from data_operation import calculate_variance, mean_squared_error
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


# Class that represents a regressor node or leaf in the regression tree
class RegressionNode():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Continuous value of node
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


class RegressionTree():
    def __init__(self, min_samples_split=20, min_var_red=1e-4,
                 max_depth=10):
        self.root = None  # Root node in regr. tree
        self.min_samples_split = min_samples_split
        self.min_var_red = min_var_red
        self.max_depth = max_depth

    def fit(self, X, y):
        # Build tree
        self.current_depth = 0
        self.root = self._build_tree(X, y)


    def _build_tree(self, X, y, current_depth=0):

        largest_variance_reduction = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # Add y as last column of X
        X_y = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split:
            # Calculate the variance reduction for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Find points to split at as the mean of every following
                # pair of points
                x = unique_values
                split_points = [(x[i-1]+x[i])/2 for i in range(1,len(x))]

                # Iterate through all unique values of feature column i and
                # calculate the variance reduction
                for threshold in split_points:
                    Xy_1, Xy_2 = divide_on_feature(X_y, feature_i, threshold)

                    if len(Xy_1) > 0 and len(Xy_2) > 0:

                        y_1 = Xy_1[:, -1]
                        y_2 = Xy_2[:, -1]

                        var_tot = calculate_variance(np.expand_dims(y, axis=1))
                        var_1 = calculate_variance(np.expand_dims(y_1, axis=1))
                        var_2 = calculate_variance(np.expand_dims(y_2, axis=1))
                        frac_1 = len(y_1) / len(y)
                        frac_2 = len(y_2) / len(y)

                        # Calculate the variance reduction
                        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

                        # If this threshold resulted in a larger variance reduction than
                        # previously registered we save the feature index and threshold
                        # and the two sets
                        if variance_reduction > largest_variance_reduction:
                            largest_variance_reduction = variance_reduction
                            best_criteria = {
                                "feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "left_branch": Xy_1, "right_branch": Xy_2}

        # If we have any information gain to go by we build the tree deeper
        if current_depth < self.max_depth and largest_variance_reduction > self.min_var_red:
            leftX, leftY = best_sets["left_branch"][
                :, :-1], best_sets["left_branch"][:, -1]    # X - all cols. but last, y - last
            rightX, rightY = best_sets["right_branch"][
                :, :-1], best_sets["right_branch"][:, -1]    # X - all cols. but last, y - last
            true_branch = self._build_tree(leftX, leftY, current_depth + 1)
            false_branch = self._build_tree(rightX, rightY, current_depth + 1)
            return RegressionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # Set y prediction for this leaf as the mean
        # of the y training data values of this leaf
        return RegressionNode(value=np.mean(y))

    # Do a recursive search down the tree and label the data sample by the
    # value of the leaf that we end up at
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # If we have a label => classify
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    # Classify samples one by one and return the set of labels
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print tree.value
        # Go deeper down the tree
        else:
            # Print test
            print "%s:%s? " % (tree.feature_i, tree.threshold)
            # Print the true scenario
            print "%sT->" % (indent),
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print "%sF->" % (indent),
            self.print_tree(tree.false_branch, indent + indent)


def main():

    X, y = datasets.make_regression(n_features=1, n_samples=100, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.3)

    clf = RegressionTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print the mean squared error
    print "Mean Squared Error:", mean_squared_error(y_test, y_pred)

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.show()


if __name__ == "__main__":
    main()
