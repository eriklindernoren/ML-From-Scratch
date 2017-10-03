
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.supervised_learning import NeuroEvolution
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy


def main():

    optimizer = Adam()

    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=4, n_clusters_per_class=1, n_informative=2)

    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = NeuroEvolution(population_size=100, 
                        mutation_rate=0.05, 
                        optimizer=optimizer, 
                        loss=CrossEntropy)
    
    model = model.evolve(X_train, y_train, n_generations=300)

    print ("Fitness: %.5f" % model.fitness)
    print ("Accuracy: %.1f%%" % float(100*model.accuracy))

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(model.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="Evolutionary Evolved Neural Network", accuracy=model.accuracy, legend_labels=range(y.shape[1]))


if __name__ == "__main__":
    main()