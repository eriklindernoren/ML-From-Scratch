from __future__ import division
import numpy as np
from sklearn import datasets
import sys, os
# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import calculate_entropy, divide_on_feature, train_test_split, accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

# Class that represents a decision node or leaf in the decision tree
class DecisionNode():
	def __init__(self, feature_i=None, threshold=None, label=None, true_branch=None, false_branch=None):
		self.feature_i = feature_i			# Index for the feature that is tested
		self.threshold = threshold			# Threshold value for feature
		self.label = label					# Label if the node is a leaf in the tree
		self.true_branch = true_branch		# 'Left' subtree
		self.false_branch = false_branch	# 'Right' subtree


class DecisionTree():
	def __init__(self):
		# Root node in the tree
		self.root = None

	def fit(self, X, y):
		# Build tree
		self.root = self._build_tree(X, y)

	def _build_tree(self, X, y):
		# Calculate the entropy by the label values
		entropy = calculate_entropy(y)

		# Save the best informaion gain
		highest_info_gain = 0
		best_criteria = None	# Feature index and threshold
		best_sets = None		# Subsets of the data

		# Add y as last column of X
		X_y = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

		n_features = np.shape(X)[1]
		n_samples = np.shape(X)[0]

		# Calculate the information gain for each feature
		for feature_i in range(n_features):
			# All values of feature_i
			feature_values = np.expand_dims(X[:, feature_i], axis=1)
			unique_values = np.unique(feature_values)

			# Iterate through all unique values of feature column i and
			# calculate the informaion gain
			for threshold in unique_values:
				Xy_1, Xy_2 = divide_on_feature(X_y, feature_i, threshold)

				# If one subset there is no use of calculating the information gain
				if len(Xy_1) > 0 and len(Xy_2) > 0:
					# Calculate information gain
					p = len(Xy_1) / n_samples
					y1 = Xy_1[:,-1]
					y2 = Xy_2[:,-1]
					info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

					# If this threshold resulted in a higher information gain than previously
					# recorded save the threshold value and the feature index
					if info_gain > highest_info_gain:
						highest_info_gain = info_gain
						best_criteria = {"feature_i": feature_i, "threshold": threshold}
						best_sets = np.array([Xy_1, Xy_2])

		# If we have any information gain to go by we build the tree deeper
		if highest_info_gain > 0:
			X_1, y_1 = best_sets[0][:, :-1], best_sets[0][:, -1]
			X_2, y_2 = best_sets[1][:, :-1], best_sets[1][:, -1]
			true_branch = self._build_tree(X_1, y_1)
			false_branch = self._build_tree(X_2, y_2)
			return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"], true_branch=true_branch, false_branch=false_branch)
		# There's no recorded information gain so we are at a leaf
		else:
			most_common = None
			max_count = 0
			results = {}
			for label in np.unique(y):
				count = len(y[y == label])
				if count > max_count:
					most_common = label
					max_count = count
			return DecisionNode(label=most_common)

	# Do a recursive search down the tree and label the data sample by the 
	# value of the leaf that we end up at
	def classify_sample(self, x, tree=None):
		if tree == None:
			tree = self.root

		# If we have a label => classify
		if tree.label != None:
			return tree.label
		
		# Choose the feature that we will test
		feature_value = x[tree.feature_i]

		# Determine if we will follow left or right branch
		branch = tree.false_branch
		if isinstance(feature_value, int) or isinstance(feature_value, float): # Interval
			if feature_value >= tree.threshold:
				branch = tree.true_branch
		elif feature_value == tree.threshold: # Nominal
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
		if tree.label!=None:
			print tree.label
		# Go deeper down the tree
		else:
			# Print test
			print "%s:%s? " % (tree.feature_i, tree.threshold)
			# Print the true scenario
			print "%sT->" % (indent),
			self.print_tree(tree.true_branch, indent+indent)
			# Print the false scenario
			print "%sF->" % (indent),
			self.print_tree(tree.false_branch, indent+indent)

# Demo of decision tree
def main():

	data = datasets.load_iris()
	X = data.data
	y = data.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

	clf = DecisionTree()
	clf.fit(X_train, y_train)
	clf.print_tree()
	y_pred = clf.predict(X_test)

	print "Accuracy:", accuracy_score(y_test, y_pred)

	pca = PCA()
	pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__": main()

