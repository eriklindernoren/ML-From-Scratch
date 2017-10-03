from __future__ import print_function, division
import numpy as np
import copy

from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.deep_learning.layers import Activation, Dense


class NeuroEvolution():
    """ Evolutionary optimization of Neural Networks.

    Parameters:
    -----------
    n_individuals: int
        The number of neural networks that are allowed in the population at a time.
    mutation_rate: float
        The probability that a weight will be mutated.
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    """
    def __init__(self, population_size, mutation_rate, optimizer, loss):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.optimizer = optimizer
        self.loss_function = loss

    def _build_mlp(self, id):
        model = NeuralNetwork(optimizer=self.optimizer, loss=self.loss_function)
        model.add(Dense(16, input_shape=(self.X.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dense(self.y.shape[1]))
        model.add(Activation('softmax'))
        
        model.id = id
        model.fitness = -1
        model.acc = -1

        return model

    def _initialize_population(self):
        """ Initialization of the neural networks forming the population"""
        self.population = []
        for _ in range(self.population_size):
            model = self._build_mlp(id=np.random.randint(1000))
            self.population.append(model)

    def _mutate(self, individual, var=1):
        """ Add zero mean gaussian noise to the layer weights with probability mutation_rate """
        for layer in individual.layers:
            if hasattr(layer, 'W'):
                # Mutation of weight with probability self.mutation_rate
                mutation_mask = np.random.binomial(1, self.mutation_rate, size=layer.W.shape)
                layer.W += np.random.normal(0, var, size=layer.W.shape) * mutation_mask
                mutation_mask = np.random.binomial(1, self.mutation_rate, size=layer.w0.shape)
                layer.w0 += np.random.normal(0, var, size=layer.w0.shape) * mutation_mask
        return individual

    def _assign_weights(self, child, parent):
        """ Copies the weights from parent to child """
        for i in range(len(child.layers)):
            if hasattr(child.layers[i], 'W'):
                child.layers[i].W = parent.layers[i].W.copy()
                child.layers[i].w0 = parent.layers[i].w0.copy()

    def _crossover(self, parent1, parent2):
        """ Performs crossover between the neurons in parent1 and parent2 to for offspring """
        child1 = self._build_mlp(id=parent1.id+1)
        self._assign_weights(child1, parent1)
        child2 = self._build_mlp(id=parent2.id+1)
        self._assign_weights(child2, parent2)

        # Perform crossover
        for i in range(len(child1.layers)):
            if hasattr(child1.layers[i], 'W'):
                n_neurons = child1.layers[i].W.shape[1]
                # Perform crossover between the individuals' neuron weights
                cutoff = np.random.randint(0, n_neurons)
                child1.layers[i].W[:, cutoff:] = parent2.layers[i].W[:, cutoff:].copy()
                child1.layers[i].w0[:, cutoff:] = parent2.layers[i].w0[:, cutoff:].copy()
                child2.layers[i].W[:, cutoff:] = parent1.layers[i].W[:, cutoff:].copy()
                child2.layers[i].w0[:, cutoff:] = parent1.layers[i].w0[:, cutoff:].copy()

        return child1, child2

    def _calculate_fitness(self):
        """ Evaluate the NNs on the test set to get fitness scores """
        for i, individual in enumerate(self.population):
            loss, acc = individual.test_on_batch(self.X, self.y)
            
            individual.fitness = 1 / (loss + 1e-8)
            individual.accuracy = acc


    def evolve(self, X, y, n_generations):
        """ Will evolve the population for n_generations based on dataset X and labels y"""
        self.X, self.y = X, y

        self._initialize_population()

        print ()
        self.population[0].summary()

        # The 40% highest fittest individuals are selected for the next generation
        n_winners = int(self.population_size * 0.4)

        for epoch in range(n_generations):
            self._calculate_fitness()

            # Sort population by fitness
            fitness_sort = np.argsort([model.fitness for model in self.population])[::-1]
            self.population = [self.population[i] for i in fitness_sort]

            # Get the individual with the highest fitness
            fittest_individual = self.population[0]
            print ("[%d Top Individual - Fitness: %.5f, Acc: %.2f%%]" % (epoch, 
                                                                        fittest_individual.fitness, 
                                                                        fittest_individual.accuracy))

            # The 'winners' are selected for the next generation
            next_population = [self.population[i] for i in range(n_winners)]

            # The rest are generated as offspring by combining the fittest individuals
            parents = [self.population[i] for i in range(self.population_size - n_winners)]
            for i in np.arange(0, len(parents), 2):
                # Perform crossover to produce offspring
                child1, child2 = self._crossover(parents[i], parents[i+1])
                # Save mutated offspring for next population
                next_population += [self._mutate(child1), self._mutate(child2)]

            self.population = next_population

        return fittest_individual

