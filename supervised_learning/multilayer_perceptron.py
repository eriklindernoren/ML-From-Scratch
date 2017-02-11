from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import train_test_split, accuracy_score, categorical_to_binary
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x)*(1-sigmoid(x))

data = datasets.load_iris()
X = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

y_train = categorical_to_binary(y_train)
y_test = categorical_to_binary(y_test)

x_train = normalize(x_train)
x_test = normalize(x_test)

# Configuration
n_hidden = 10
n_iterations = 10000
n_samples = len(x_train)
n_features = len(x_train[0])
learning_rate = 0.01

# Initial weights between [-1/sqrt(N), 1/sqrt(N)] (w - hidden, v - output)
a = -1/math.sqrt(n_features)
b = -a
w = (b-a)*np.random.random((len(x_train[0]), n_hidden)) + a
v = (b-a)*np.random.random((n_hidden, len(y_train[0,:]))) + a

errors = []
for i in range(n_iterations):
	# Calculate outputs of hidden layer
	hidden_input = x_train.dot(w)
	hidden_output = sigmoid(hidden_input)
	# Calculate outputs
	output_layer_input = hidden_output.dot(v)
	output_layer_pred = sigmoid(output_layer_input)
	
	mean_squared_error = np.mean(np.power(y_train - output_layer_pred, 2))
	errors.append(mean_squared_error)
	
	# Calculate the loss gradient
	v_gradient = -2*(y_train - output_layer_pred)*sigmoid_gradient(output_layer_input)
	w_gradient = v_gradient.dot(v.T)*sigmoid_gradient(hidden_input)

	# Update weights
	v -= learning_rate*hidden_output.T.dot(v_gradient)
	w -= learning_rate*x_train.T.dot(w_gradient)

# Plot the training error
plt.plot(range(n_iterations), errors)
plt.ylabel('Training Error')
plt.xlabel('Iterations')
plt.show()

# Predict x_test
# Calculate outputs of hidden layer
hidden_output = sigmoid(np.dot(x_test,w))
y_pred = np.round(sigmoid(np.dot(hidden_output, v)))

# Print prediction and true output
print y_pred
print y_test
print "Accuracy:", accuracy_score(y_test, y_pred)

