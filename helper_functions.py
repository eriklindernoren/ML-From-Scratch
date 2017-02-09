from __future__ import division
import numpy as np

def train_test_split(X, Y, test_size=0.5, shuffle=True):
	if shuffle:
		# Concatenate x and y and do a random shuffle
		x_y = np.concatenate((X,Y.reshape((1,len(Y))).T), axis=1)
		np.random.shuffle(x_y)
		X = x_y[:,:-1]
		Y = x_y[:,-1].astype(int)
	# Split the training data from test data in the ration specified in test_size
	split_i = len(Y) - int(len(Y)//(1/test_size))
	x_train = X[:split_i]
	y_train = Y[:split_i]
	x_test = X[split_i:]
	y_test = Y[split_i:]

	return x_train, x_test, y_train, y_test

def accuracy_score(y_true, y_pred):
	correct = 0
	for i in range(len(y_true)):
		eq = np.equal(y_true[i], y_pred[i])
		if isinstance(eq, np.bool_):
			if eq == False:
				continue
		elif False in eq:
			continue
		correct += 1

	return correct / len(y_true)
