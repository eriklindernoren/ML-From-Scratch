from sklearn import datasets
import sys, os, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, categorical_to_binary, normalize, binary_to_categorical
from data_operation import accuracy_score
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Gradient of activation function
def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))


class MultilayerPerceptron():
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden    # Number of hidden neurons
        self.W = None               # Hidden layer weights
        self.V = None               # Output layer weights

    def fit(self, X, y, n_iterations=3000, learning_rate=0.01, plot_errors=False):
        X_train = np.array(X, dtype=float)
        # Convert the nominal y values to binary
        y_train = categorical_to_binary(y)

        # Insert dummy values for bias weights W0
        X_train = np.insert(X_train, 0, 1, axis=1)

        n_samples = np.shape(X_train)[0]
        n_features = np.shape(X_train)[1]
        n_outputs = np.shape(y_train)[1]

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)]
        a = -1/math.sqrt(n_features)
        b = -a
        self.W = (b-a)*np.random.random((n_features, self.n_hidden)) + a
        self.V = (b-a)*np.random.random((self.n_hidden+1, n_outputs)) + a

        errors = []
        for i in range(n_iterations):
            # Calculate hidden layer
            hidden_input = X_train.dot(self.W)
            # Calculate output of hidden neurons and add dummy values for bias weights V0
            hidden_output = np.insert(sigmoid(hidden_input), 0, 1, axis=1)
            
            # Calculate output layer
            output_layer_input = hidden_output.dot(self.V)
            output_layer_pred = sigmoid(output_layer_input)

            # Calculate the error
            mean_squared_error = np.mean(np.power(y_train - output_layer_pred, 2))
            errors.append(mean_squared_error)

            # Calculate loss gradients:
            # Output layer weights V
            v_gradient = -2*(y_train - output_layer_pred)*sigmoid_gradient(output_layer_input)
            # Hidden layer weights W (disregard bias weight of V when determining W)
            w_gradient = v_gradient.dot(self.V[1:,:].T)*sigmoid_gradient(hidden_input)

            # Update weights
            self.V -= learning_rate*hidden_output.T.dot(v_gradient)
            self.W -= learning_rate*X_train.T.dot(w_gradient)
        
        # Plot the training error
        if plot_errors:
            plt.plot(range(n_iterations), errors)
            plt.ylabel('Training Error')
            plt.xlabel('Iterations')
            plt.show()

    # Use the trained model to predict labels of X
    def predict(self, X):
        # Insert dummy values for bias weights W0
        X_test = np.insert(np.array(X, dtype=float), 0, 1, axis=1)
        # Insert dummy for bias weights V0
        hidden_output = np.insert(sigmoid(np.dot(X_test,self.W)), 0, 1, axis=1)
        y_pred = np.round(sigmoid(np.dot(hidden_output, self.V)))
        # Convert binary representation of y to nominal labels
        y_pred = binary_to_categorical(y_pred)
        return y_pred

# Demo
def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # MLP
    clf = MultilayerPerceptron(n_hidden=10)
    clf.fit(X_train, y_train, n_iterations=4000, learning_rate=0.01)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)
    

if __name__ == "__main__": main()
