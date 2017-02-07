import pandas as pd
import numpy as np
import math

# The sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x)*(1-sigmoid(x))

# test predictions
dataset = np.array([[1, 2.7810836,2.550537003,0],
	[1, 1.465489372,2.362125076,0],
	[1, 3.396561688,4.400293529,0],
	[1, 1.38807019,1.850220317,0],
	[1, 3.06407232,3.005305973,0],
	[1, 7.627531214,2.759262235,1],
	[1, 5.332441248,2.088626775,1],
	[1, 6.922596716,1.77106367,1],
	[1, 8.675418651,-0.242068655,1],
	[1, 7.673756466,3.508563011,1]])

x_train = dataset[:,0:-1]
y_train = dataset[:,-1].reshape((len(dataset[:,-1]),1))

n_features = len(x_train[0])
n_iterations = 3000
learning_rate = 0.001

# Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
a = -1/math.sqrt(n_features)
b = -a
parameters = (b-a)*np.random.random((len(x_train[0]), 1)) + a

for i in range(n_iterations):
	dot = x_train.dot(parameters)
	y_pred = sigmoid(dot)

	# Calculate the loss gradient
	loss_gradient = -2*(y_train - y_pred)*sigmoid_gradient(dot)

	# Update weights
	parameters -= learning_rate*x_train.T.dot(loss_gradient)

print "Y prediction:"
print y_pred
print "Y:"
print y_train



