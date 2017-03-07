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
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is nominal
        self.one_dim = None

        # If Gradient Boost
        self.loss = None
        self.xgboost = False

    def fit(self, X, y, loss=None):
        # Build tree
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

        self.loss=None

    def _build_tree(self, X, y, current_depth=0):

        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        expand_needed = len(np.shape(y)) == 1
        if expand_needed:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        X_y = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    Xy_1, Xy_2 = divide_on_feature(X_y, feature_i, threshold)
                    
                    if len(Xy_1) > 0 and len(Xy_2) > 0:
                        y_1 = Xy_1[:, n_features:]
                        y_2 = Xy_2[:, n_features:]

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


        if largest_impurity > self.min_impurity:
            leftX = best_sets["left_branch"][:, :n_features]
            leftY = best_sets["left_branch"][:, n_features:]    # X - all cols. but last, y - last
            rightX = best_sets["right_branch"][:, :n_features]
            rightY = best_sets["right_branch"][:, n_features:]    # X - all cols. but last, y - last
            true_branch = self._build_tree(leftX, leftY, current_depth + 1)
            false_branch = self._build_tree(rightX, rightY, current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    # Do a recursive search down the tree and make a predict of the data sample by the
    # value of the leaf that we end up at
    def predict_value(self, x, tree=None):
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



class XGBoostRegressionTree(DecisionTree):

    # For XGBoost
    # http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

    def _gain(self, y, y_pred):
        nom = np.power(self.loss.gradient(y, y_pred).sum(), 2)
        denom = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nom / denom)

    def _gain_by_taylor(self, y, y_1, y_2):
        # y_true left part, y_pred right part
        split = int(np.shape(y)[1]/2)
        y_pred = y[:, split:]
        y = y[:, :split]
        y_1_pred = y_1[:, split:]
        y_1 = y_1[:, :split]
        y_2_pred = y_2[:, split:]
        y_2 = y_2[:, :split]

        true_gain = self._gain(y_1, y_1_pred)
        false_gain = self._gain(y_2, y_2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approx(self, y):
        split = int(np.shape(y)[1]/2)
        y_pred = y[:, split:]
        y = y[:, :split]
        value = np.sum(self.loss.gradient(y, y_pred), axis=0) / np.sum(self.loss.hess(y, y_pred).sum(), axis=0)
        return value

    def fit(self, X, y, loss):
        self.loss = loss
        self.xgboost = True
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approx
        super(XGBoostRegressionTree, self).fit(X, y)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y, y_1, y_2):

        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y_1)
        var_2 = calculate_variance(y_2)
        frac_1 = len(y_1) / len(y)
        frac_2 = len(y_2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    # For XGBoost
    # http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
    def _gain_by_taylor(self, y, y_1, y_2):
        # y_true left part, y_pred right part
        split = int(np.shape(y)[1]/2)
        y_pred = y[:, split:]
        y = y[:, :split]
        y_1_pred = y_1[:, split:]
        y_1 = y_1[:, :split]
        y_2_pred = y_2[:, split:]
        y_2 = y_2[:, :split]


        true_gain = self.loss.gain(y_1, y_1_pred)
        false_gain = self.loss.gain(y_2, y_2_pred)
        gain = self.loss.gain(y, y_pred)
        return true_gain + false_gain - gain

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        value = value if len(value) > 1 else value[0]
        return value

    def fit(self, X, y, loss=None):
        self._impurity_calculation = self._calculate_variance_reduction
        self.loss = loss
        if self.loss:
            self.xgboost = True
            self._impurity_calculation = self._gain_by_taylor
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=4)

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
