from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import divide_on_feature
from data_manipulation import train_test_split, standardize
from data_operation import calculate_entropy, accuracy_score
from data_operation import mean_squared_error, calculate_variance
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


# Class that represents a decision node or leaf in the decision tree
class DecisionNode():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf")):
        self.root = None  # Root node in dec. tree
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self._leaf_value_calculation = None

    def fit(self, X, y):
        # Build tree
        self.current_depth = 0
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):

        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # Add y as last column of X
        X_y = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    Xy_1, Xy_2 = divide_on_feature(X_y, feature_i, threshold)
                    # If one subset there is no use of calculating the
                    # information gain
                    if len(Xy_1) > 0 and len(Xy_2) > 0:
                        y_1 = Xy_1[:, -1]
                        y_2 = Xy_2[:, -1]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y_1, y_2)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "left_branch": Xy_1, "right_branch": Xy_2}

        # If we have any information gain to go by we build the tree deeper
        if self.current_depth < self.max_depth and largest_impurity > self.min_impurity:
            leftX, leftY = best_sets["left_branch"][
                :, :-1], best_sets["left_branch"][:, -1]    # X - all cols. but last, y - last
            rightX, rightY = best_sets["right_branch"][
                :, :-1], best_sets["right_branch"][:, -1]    # X - all cols. but last, y - last
            true_branch = self._build_tree(leftX, leftY)
            false_branch = self._build_tree(rightX, rightY)
            self.current_depth += 1
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    # Do a recursive search down the tree and make a predict of the data sample by the
    # value of the leaf that we end up at
    def classify_sample(self, x, tree=None):
        if tree is None:
            tree = self.root

        # If we have a value => return prediction
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
        return self.classify_sample(x, branch)

    # Classify samples one by one and return the set of labels
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.classify_sample(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y, y_1, y_2):
        var_tot = calculate_variance(np.expand_dims(y, axis=1))
        var_1 = calculate_variance(np.expand_dims(y_1, axis=1))
        var_2 = calculate_variance(np.expand_dims(y_2, axis=1))
        frac_1 = len(y_1) / len(y)
        frac_2 = len(y_2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return variance_reduction

    def _mean_of_y(self, y):
        return np.mean(y)

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y_1, y_2):
        # Calculate information gain
        p = len(y_1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y_1) - (1 - p) * \
            calculate_entropy(y_2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        results = {}
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count

        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


def main():

    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy:", accuracy_score(y_test, y_pred))

    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)

    print ("-- Regression Tree --")

    X, y = datasets.make_regression(n_features=1, n_samples=100, bias=0, noise=5)

    X_train, X_test, y_train, y_test = train_test_split(standardize(X), y, test_size=0.3)

    clf = RegressionTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    print ("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Plot the results
    plt.scatter(X_test[:, 0], y_test, color='black')
    plt.scatter(X_test[:, 0], y_pred, color='green')
    plt.show()


if __name__ == "__main__":
    main()
