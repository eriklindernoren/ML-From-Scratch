from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math

x_train = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y_train = np.array([[0,0,0],[0,1,0],[0,1,0],[0,0,1]])

x_test = np.array([[1,1,1],[0,0,1],[1,0,1]])
y_test = np.array([[0,0,1],[0,0,0],[0,1,0]])

# Configuration
n_hidden = 7
n_iterations = 10000
n_samples = len(x_train)
learning_rate = 0.1

# Initial weights (w - hidden, v - output)
w = 2*np.random.random((n_hidden, len(x_train[0]))) - 1
v = 2*np.random.random((len(y_train[0,:]), n_hidden)) - 1

errors = []
for i in range(n_iterations):
	# Calculate outputs of hidden layer
	l1 = 1/(1+np.exp(-(np.dot(x_train,w.T))))
	# Calculate outputs
	l2 = 1/(1+np.exp(-(np.dot(l1,v.T))))
	
	error = (y_train - l2)
	errors.append(math.fabs(error.mean()))
	
	# Calculate the gradient of the error
	l2_delta = -(y_train - l2)*(l2*(1-l2))
	l1_delta = l2_delta.dot(v)*(l1*(1-l1))

	# Update weights
	v -= learning_rate*l2_delta.T.dot(l1)
	w -= learning_rate*l1_delta.T.dot(x_train)


# Plot the training error
plt.plot(range(n_iterations), errors)
plt.ylabel('Training Error')
plt.xlabel('Iterations')
plt.show()

# Predict x_test
l1 = 1/(1+np.exp(-(np.dot(x_test,w.T))))
y_pred = np.round(1/(1+np.exp(-(np.dot(l1,v.T)))))

# Print prediction and true output
print np.round(y_pred)
print y_test
print "Accuracy:", accuracy_score(y_test, y_pred)

