from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x)*(1-sigmoid(x))

def categorical_to_binary(x):
	n_col = np.amax(x)+1
	binarized = np.zeros((len(x), n_col))
	for i in range(len(x)):
		binarized[i, x[i]] = 1
	return binarized	

iris = datasets.load_digits()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_train = categorical_to_binary(y_train)
y_test = categorical_to_binary(y_test)

# Configuration
n_hidden = 60
n_iterations = 12000
n_samples = len(x_train)
learning_rate = 0.001

# Initial weights (w - hidden, v - output)
w = 2*np.random.random((len(x_train[0]), n_hidden)) - 1
v = 2*np.random.random((n_hidden, len(y_train[0,:]))) - 1

errors = []
for i in range(n_iterations):
	# Calculate outputs of hidden layer
	hidden_input = np.dot(x_train,w)
	hidden_output = sigmoid(hidden_input)
	# Calculate outputs
	output_layer_input = np.dot(hidden_output, v)
	output = sigmoid(output_layer_input)
	
	error = (y_train - output)
	errors.append(math.fabs(error.mean()))
	
	# Calculate the loss gradient
	l2_delta = -(y_train - output)*sigmoid_gradient(output_layer_input)
	l1_delta = l2_delta.dot(v.T)*sigmoid_gradient(hidden_input)

	# Update weights
	v -= learning_rate*hidden_output.T.dot(l2_delta)
	w -= learning_rate*x_train.T.dot(l1_delta)


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

