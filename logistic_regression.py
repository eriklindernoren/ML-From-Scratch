import numpy as np
import math

# The sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x)*(1-sigmoid(x))

def make_diagonal(x):
	m = np.zeros((len(x), len(x)))
	for i in range(len(m[0])):
		m[i,i] = x[i]
	return m

# Data
dataset = np.array([[1,2.7810836,2.550537003,		0],
					[1,1.465489372,2.362125076,		0],
					[1,3.396561688,4.400293529,		0],
					[1,1.38807019,1.850220317,		0],
					[1,3.06407232,3.005305973,		0],
					[1,7.627531214,2.759262235,		1],
					[1,5.332441248,2.088626775,		1],
					[1,6.922596716,1.77106367,		1],
					[1,8.675418651,-0.242068655,	1],
					[1,7.673756466,3.508563011,		1]
					])

x_train = dataset[:,0:-1]
y_train = dataset[:,-1].reshape((len(dataset[:,-1]),1))

n_features = len(x_train[0])
n_iterations = 5

# Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
a = -1/math.sqrt(n_features)
b = -a
param = (b-a)*np.random.random((len(x_train[0]), 1)) + a

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

# Print the last prediction
print "Y prediction:"
print y_pred
print "Y:"
print y_train



