import sys, os, math
from sklearn import datasets
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


class Perceptron():
    def __init__(self):
        # Weights
        self.W = None

    def fit(self, X, y, n_iterations=40000, learning_rate=0.01, plot_errors=False):
        X_train = np.array(X, dtype=float)
        # Convert the nominal y values to binary
        y_train = categorical_to_binary(y)

        n_neurons = len(y_train[0,:])
        n_samples = len(X_train)
        n_features = len(X_train[0])

        # Initial weights between [-1/sqrt(N), 1/sqrt(N)]
        a = -1/math.sqrt(n_features)
        b = -a
        self.W = (b-a)*np.random.random((len(X_train[0]), n_neurons)) + a

        errors = []
        for i in range(n_iterations):
            # Calculate outputs
            neuron_input = np.dot(X_train,self.W)
            neuron_output = sigmoid(neuron_input)
            
            mean_squared_error = np.mean(np.power(y_train - neuron_output, 2))
            errors.append(mean_squared_error)
            
            # Calculate the loss gradient
            w_gradient = -2*(y_train - neuron_output)*sigmoid_gradient(neuron_input)

            # Update weights
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
        y_pred = np.round(sigmoid(np.dot(X_test,self.W)))
        y_pred = binary_to_categorical(y_pred)
        return y_pred

# Demo
def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Perceptron
    clf = Perceptron()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print "Accuracy:", accuracy_score(y_test, y_pred)

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)
    

if __name__ == "__main__": main()
