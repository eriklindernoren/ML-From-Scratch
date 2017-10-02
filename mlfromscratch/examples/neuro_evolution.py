
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.supervised_learning import NeuroEvolution
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.utils import get_random_subsets, shuffle_data, accuracy_score
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy, SquareLoss
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation


def main():

    optimizer = Adam()

    #-----
    # MLP
    #-----

    X, y = datasets.make_classification(n_samples=1000, n_features=4)

    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    n_samples, n_features = X.shape
    n_hidden = 512

    model = NeuroEvolution(population_size=10, 
                        mutation_rate=0.01, 
                        n_parents=4, 
                        recombination_rate=0.1, 
                        optimizer=optimizer, 
                        loss=CrossEntropy)
    
    model, fitness, acc = model.evolve(X_train, y_train, n_generations=1000)
    


if __name__ == "__main__":
    main()