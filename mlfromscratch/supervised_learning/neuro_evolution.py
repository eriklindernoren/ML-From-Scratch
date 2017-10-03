from __future__ import print_function, division
import numpy as np
import copy

from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.deep_learning.layers import Activation, Dense


class NeuroEvolution():
    """ Evolutionary Neural Network optimization.

    Parameters:
    -----------
    n_individuals: int
        The number of neural networks that are allowed in the population at a time.
    mutation_rate: float
        The probability that a weight will be mutated.
    n_parents: int
        The number of parents that will be selected to form offspring each generation.
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    """
    def __init__(self, population_size, mutation_rate, n_parents, optimizer, loss):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_parents = n_parents
        self.optimizer = optimizer
        self.loss_function = loss

    def _initialize_population(self, n_features, n_outputs):
        """ Initialization of the neural networks forming the population"""
        self.population = []
        for _ in range(self.population_size):
            model = NeuralNetwork(optimizer=self.optimizer, loss=self.loss_function)
            model.add(Dense(16, input_shape=(n_features,)))
            model.add(Activation('relu'))
            model.add(Dense(n_outputs))
            model.add(Activation('softmax'))

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

    def _recombination(self, parent1, parent2):
        """ Swap the weights of parent1 and parent2 to form new children """
        child1 = copy.copy(parent1)
        child2 = copy.copy(parent2)
        for i in range(len(child1.layers)):
            if hasattr(child1.layers[i], 'W'):
                # Perform crossover between the individuals' neuron weights
                cutoff = np.random.randint(0, child1.layers[i].W.shape[1])
                child1.layers[i].W[:, cutoff:] = parent2.layers[i].W[:, cutoff:].copy()
                child1.layers[i].w0[:, cutoff:] = parent2.layers[i].w0[:, cutoff:].copy()
                child2.layers[i].W[:, cutoff:] = parent1.layers[i].W[:, cutoff:].copy()
                child2.layers[i].w0[:, cutoff:] = parent1.layers[i].w0[:, cutoff:].copy()

        return child1, child2

    def _calculate_fitness(self, population):
        """ Evaluate the NNs on the test set to get fitness scores """
        population_fitness = np.empty(len(population))
        accuracies = np.empty_like(population_fitness)
        for i, individual in enumerate(population):
            loss, acc = individual.test_on_batch(self.X, self.y)
            fitness = 1 / (loss + 1e-8)
            population_fitness[i] = fitness
            accuracies[i] = acc

        return population_fitness, accuracies


    def evolve(self, X, y, n_generations):
        """ Will evolve the population for n_generations based on dataset X and labels y"""
        self.X, self.y = X, y

        self._initialize_population(n_features=X.shape[1], n_outputs=y.shape[1])

        # The 40% highest fittest individuals are selected for the next generation
        n_winners = int(self.population_size * 0.4)

        for epoch in range(n_generations):
            population_fitness, accuracies = self._calculate_fitness(self.population)

            # Get the individual with the highest fitness
            fittest_individual = self.population[np.argmax(population_fitness)]
            highest_fitness = max(population_fitness)
            highest_accuracy = max(accuracies)

            print ("[%d Closest Candidate - Fitness: %.5f, Acc: %.2f%%]" % (epoch, highest_fitness, highest_accuracy))

            # Set the probability that the individual should be selected as a parent
            # proportionate to the individual's fitness.
            parent_probabilities = [fitness / sum(population_fitness) for fitness in population_fitness]

            # Select the n_winners for next generation
            highest_i = np.argsort(population_fitness)[-n_winners:]
            next_population = [self.population[cand_i] for cand_i in highest_i]

            population_candidates = []
            for _ in np.arange(0, self.population_size, 2):
                # Select two parents randomly according to parent selection probabilities
                parent1, parent2 = np.random.choice(self.population, size=2, p=parent_probabilities, replace=False)
                # Perform crossover to produce offspring
                child1, child2 = self._recombination(parent1, parent2)
                # Save mutated offspring as candidates for next population
                population_candidates += [self._mutate(child1), self._mutate(child2)]

            # Select individuals for next generation based on fitness
            population_fitness, accuracies = self._calculate_fitness(population_candidates)
            selection_probabilities = [fitness / sum(population_fitness) for fitness in population_fitness]
            next_population += np.random.choice(population_candidates, 
                                            size=self.population_size-n_winners, 
                                            p=selection_probabilities,
                                            replace=True).tolist()

            self.population = next_population

        return fittest_individual, highest_fitness, highest_accuracy

