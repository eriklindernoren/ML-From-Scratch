import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
x_train = X[:-20]
x_test = X[-20:]

# Split the targets into training/testing sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# Insert constant 1:s for bias weight
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# Get weights by least squares
w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)

# Make prediction
y_pred = x_test.dot(w)

# Print the mean squared error
mean_squared_error = np.mean(np.power(y_test - y_pred, 2))
print "Mean Squared Error:", mean_squared_error

# Plot the results
plt.scatter(x_test[:,1], y_test,  color='black')
plt.plot(x_test[:,1], y_pred, color='blue', linewidth=3)
plt.show()
