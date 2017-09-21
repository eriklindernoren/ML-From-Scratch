
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.utils import get_random_subsets, shuffle_data, accuracy_score
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation


def main():

    optimizer = Adam()

    #-----
    # MLP
    #-----

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    n_samples, n_features = X.shape
    n_hidden = 512

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('leaky_relu'))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="MLP")
    
    train_err, val_err = clf.fit(X_train, y_train, n_epochs=50, batch_size=256)
    
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=range(10))


if __name__ == "__main__":
    main()