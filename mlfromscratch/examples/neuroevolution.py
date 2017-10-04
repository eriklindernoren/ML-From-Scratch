
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from mlfromscratch.supervised_learning import Neuroevolution
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.deep_learning.layers import Activation, Dense
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.deep_learning.optimizers import Adam

def main():

    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=4, n_clusters_per_class=1, n_informative=2)

    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    y = to_categorical(y.astype("int"))

    # Model builder
    def model_builder(n_inputs, n_outputs):    
        model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
        model.add(Dense(16, input_shape=(n_inputs,)))
        model.add(Activation('relu'))
        model.add(Dense(n_outputs))
        model.add(Activation('softmax'))

        return model

    # Print the model summary of a individual in the population
    print ()
    model_builder(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary()

    population_size = 100
    n_generations = 1000
    mutation_rate = 0.01

    print ("Population Size: %d" % population_size)
    print ("Generations: %d" % n_generations)
    print ("Mutation Rate: %d" % mutation_rate)
    print ("")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    model = Neuroevolution(population_size=population_size, 
                        mutation_rate=mutation_rate, 
                        model_builder=model_builder)
    
    model = model.evolve(X_train, y_train, n_generations=n_generations)

    loss, accuracy = model.test_on_batch(X_test, y_test)
    print ("Test set accuracy: %.1f%%" % float(100*accuracy))

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(model.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="Evolutionary Evolved Neural Network", accuracy=accuracy, legend_labels=range(y.shape[1]))


if __name__ == "__main__":
    main()