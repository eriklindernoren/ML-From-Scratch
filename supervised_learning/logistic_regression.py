from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')
from helper_functions import accuracy_score, make_diagonal
import numpy as np
import pandas as pd
import math, sys

# The sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x)*(1-sigmoid(x))

df = pd.read_csv("./data/diabetes.csv")

Y = df["Outcome"]
y_train = Y[:-300].as_matrix()
y_test = Y[-300:].as_matrix()

X = df.drop("Outcome", axis=1)
x_train = np.insert(normalize(X[:-300].as_matrix()),0,1,axis=1)
x_test = np.insert(normalize(X[-300:].as_matrix()),0,1,axis=1)

n_features = len(x_train[0])
n_iterations = 10

# Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
a = -1/math.sqrt(n_features)
b = -a
param = (b-a)*np.random.random((len(x_train[0]),)) + a


# Tune parameters for n iterations
for i in range(n_iterations):
	# Make a new prediction
	dot = x_train.dot(param)
	y_pred = sigmoid(dot)

	# Make a diagonal matrix of the sigmoid gradient column vector
	diag_gradient = make_diagonal(sigmoid_gradient(dot))

	# Batch opt:
	# (X^T * diag(sigm*(1 - sigm) * X) * X^T * (diag(sigm*(1 - sigm) * X * param + Y - Y_pred)
	param = np.linalg.inv(x_train.T.dot(diag_gradient).dot(x_train)).dot(x_train.T).dot(diag_gradient.dot(x_train).dot(param) + y_train - y_pred)

# Print a final prediction
dot = x_test.dot(param)
y_pred = np.round(sigmoid(dot)).astype(int)
print "Y prediction:"
print y_pred
print "Y:"
print y_test

print "Accuracy:", accuracy_score(y_test, y_pred)

