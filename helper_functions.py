from __future__ import division
import numpy as np
import math, sys

def shuffle_data(X, y):
	# Concatenate x and y and do a random shuffle
	x_y = np.concatenate((X,y.reshape((1,len(y))).T), axis=1)
	np.random.shuffle(x_y)
	X = x_y[:,:-1] # every column except the last
	y = x_y[:,-1].astype(int) # last column

	return X, y

# Divide dataset based on if sample value on feature index is larger than
# the given threshold
def divide_on_feature(X, feature_i, threshold):
	split_func = None
	if isinstance(threshold, int) or isinstance(threshold, float):
		split_func = lambda sample: sample[feature_i] >= threshold
	else:
		split_func = lambda sample: sample[feature_i] == threshold

	X_1 = np.array([sample for sample in X if split_func(sample)])
	X_2 = np.array([sample for sample in X if not split_func(sample)])

	return np.array([X_1, X_2])


# Calculate the entropy of label array y
def calculate_entropy(y):
	log2=lambda x:math.log(x)/math.log(2)
	# Get label as last element in dataset
	unique_labels = np.unique(y)
	entropy = 0
	for label in unique_labels:
		count = len(y[y == label])
		p = count / len(y)
		entropy += -p*log2(p)
	return entropy

# Split the data into train and test sets
def train_test_split(X, y, test_size=0.5, shuffle=True):
	if shuffle:
		X, y = shuffle_data(X, y)
	# Split the training data from test data in the ratio specified in test_size
	split_i = len(y) - int(len(y)//(1/test_size))
	x_train, x_test = X[:split_i], X[split_i:]
	y_train, y_test = y[:split_i], y[split_i:]

	return x_train, x_test, y_train, y_test

# Split the data into k sets of training / test data
def k_fold_cross_validation_sets(X, y, k, shuffle=True):
	if shuffle:
		X, y = shuffle_data(X, y)

	n_samples = len(y)
	left_overs = {}
	n_left_overs = (n_samples % k)
	if n_left_overs != 0:
		left_overs["X"] = X[-n_left_overs:]
		left_overs["y"] = y[-n_left_overs:]
		X = X[:-n_left_overs]
		y = y[:-n_left_overs]
		
	X_split = np.split(X, k)
	y_split = np.split(y, k)
	sets = []
	for i in range(k):
		X_test, y_test = X_split[i], y_split[i]
		X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=0)
		y_train = np.concatenate(y_split[:i] + y_split[i+1:], axis=0)
		sets.append([X_train, X_test, y_train, y_test])

	# Add left over samples to last set as training samples
	if n_left_overs != 0:
		np.append(sets[-1][0], left_overs["X"], axis=0)
		np.append(sets[-1][2], left_overs["y"], axis=0)

	return np.array(sets)

# Normalize the dataset X
def normalize(X, axis=-1, order=2):
	l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
	l2[l2==0] = 1
	return X / np.expand_dims(l2, axis)

# Returns the mean squared error between y_true and y_pred
def mean_squared_error(y_true, y_pred):
	mse = np.mean(np.power(y_true - y_pred, 2))
	return mse

# Return the variance of the features in dataset X
def calculate_variance(X):
	mean = np.ones(np.shape(X))*X.mean(0)
	n_samples = np.shape(X)[0]
	variance = (1/n_samples) * np.diag((X - mean).T.dot(X - mean))

	return variance

# Calculate the standard deviations of the features in dataset X
def calculate_std_dev(X):
	std_dev = np.sqrt(calculate_variance(X))

	return std_dev

# Making an array of nominal values into a binarized matrix
def categorical_to_binary(x):
	n_col = np.amax(x)+1
	binarized = np.zeros((len(x), n_col))
	for i in range(len(x)):
		binarized[i, x[i]] = 1

	return binarized

# Converting from binary vectors to nominal values
def binary_to_categorical(x):
	categorical = []
	for i in range(len(x)):
		if not 1 in x[i]:
			categorical.append(0)
		else:
			i_where_one = np.where(x[i] == 1)[0][0]
			categorical.append(i_where_one)

	return categorical

# Calculate the distance between two vectors
def euclidean_distance(x1, x2):
	distance = 0
	for i in range(len(x1)):
		distance += pow((x1[i] - x2[i]), 2)

	return math.sqrt(distance)

# Converts a vector into an diagonal matrix
def make_diagonal(x):
	m = np.zeros((len(x), len(x)))
	for i in range(len(m[0])):
		m[i,i] = x[i]

	return m

# Compare y_true to y_pred and return the accuracy
def accuracy_score(y_true, y_pred):
	correct = 0
	for i in range(len(y_true)):
		diff = y_true[i] - y_pred[i]
		if diff == np.zeros(np.shape(diff)):
			correct += 1
	return correct / len(y_true)

# Calculate the covariance matrix for the dataset X
def calculate_covariance_matrix(X, Y=None):
	if not Y:
		Y = X
	X_mean = np.ones(np.shape(X))*X.mean(0)
	Y_mean = np.ones(np.shape(Y))*Y.mean(0)
	n_samples = np.shape(X)[0]
	covariance_matrix = (1/(n_samples-1)) * (X - X_mean).T.dot(Y - Y_mean)

	return np.array(covariance_matrix, dtype=float)



# Calculate the correlation matrix for the dataset X
def calculate_correlation_matrix(X, Y=None):
	if not Y:
		Y = X
	covariance = calculate_covariance_matrix(X, Y)
	std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
	std_dev_Y = np.expand_dims(calculate_std_dev(Y), 1)
	correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

	return np.array(correlation_matrix, dtype=float)



