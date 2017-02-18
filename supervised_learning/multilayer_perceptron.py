from sklearn import datasets
import sys, os, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import train_test_split, accuracy_score, categorical_to_binary, normalize, binary_to_categorical
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
        self.n_hidden = n_hidden
        # Weights (w - hidden, v - output)
        self.W = None
        self.V = None

    def fit(self, X, y, n_iterations=3000, learning_rate=0.01, plot_errors=False):
        X_train = np.array(X, dtype=float)
        # Convert the nominal y values to binary
        y_train = categorical_to_binary(y)

        n_samples = len(X_train)
        n_features = len(X_train[0])

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)]
        a = -1/math.sqrt(n_features)
        b = -a
        self.W = (b-a)*np.random.random((len(X_train[0]), self.n_hidden)) + a
        self.V = (b-a)*np.random.random((self.n_hidden, len(y_train[0,:]))) + a

        errors = []
        for i in range(n_iterations):
            # Calculate outputs of hidden layer
            hidden_input = X_train.dot(self.W)
            hidden_output = sigmoid(hidden_input)
            # Calculate outputs
            output_layer_input = hidden_output.dot(self.V)
            output_layer_pred = sigmoid(output_layer_input)

            mean_squared_error = np.mean(np.power(y_train - output_layer_pred, 2))
            errors.append(mean_squared_error)

            # Calculate the loss gradient
            v_gradient = -2*(y_train - output_layer_pred)*sigmoid_gradient(output_layer_input)
            w_gradient = v_gradient.dot(self.V.T)*sigmoid_gradient(hidden_input)

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
        X_test = np.array(X, dtype=float)
        hidden_output = sigmoid(np.dot(X_test,self.W))
        y_pred = np.round(sigmoid(np.dot(hidden_output, self.V)))
        # Convert binary representation of y to nominal labels
        y_pred = binary_to_categorical(y_pred)
        return y_pred

# Demo
def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # MLP
    clf = MultilayerPerceptron(n_hidden=10)
    clf.fit(X_train, y_train, n_iterations=4000, learning_rate=0.01)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)
    

if __name__ == "__main__": main()
