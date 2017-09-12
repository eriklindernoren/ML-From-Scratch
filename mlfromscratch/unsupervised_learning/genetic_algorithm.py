import string
import numpy as np

class GeneticAlgorithm():
    """A implementation of a Genetic Algorithm which will try to produce the user
    specified target string.

    Parameters:
    -----------
    target_string: string
        The string which the GA should try to produce.
    population_size: int
        The number of individuals (possible solutions) in the population.
    mutation_rate: float
        The rate (or probability) of which the alleles (chars in this case) should be
        randomly changed.
    """
    def __init__(self, target_string, population_size, mutation_rate):
        self.target = target_string
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.eps = 1e-8
        self.letters = [" "] + list(string.letters)

    def _initialize(self):
        # Initialize population with random strings
        self.population = []
        for _ in range(self.population_size):
            # Select random letters as new individual
            individual = "".join(np.random.choice(self.letters, size=len(self.target)))
            self.population.append(individual)

    def _determine_fitness(self):
        """ Calculates the fitness of each individual in the population """
        population_fitness = []
        for individual in self.population:
            # Calculate loss as the alphabetical distance between
            # the characters in the individual and the target string
            loss = 0
            for i in range(len(individual)):
                letter_i1 = self.letters.index(individual[i])
                letter_i2 = self.letters.index(self.target[i])
                loss += abs(letter_i1 - letter_i2)
            fitness = 1 / (loss + self.eps)
            population_fitness.append(fitness)
        return population_fitness

    def _mutate(self, individual):
        individual = list(individual)
        for j in range(len(individual)):
            # Make change with probability mutation_rate
            if np.random.random() < self.mutation_rate:
                individual[j] = np.random.choice(self.letters)
        # Return mutated individual as string
        return "".join(individual)

    def _crossover(self, parent1, parent2):
        # Select random crossover point
        cross_i = np.random.randint(0, len(parent1))
        child1 = parent1[:cross_i] + parent2[cross_i:]
        child2 = parent2[:cross_i] + parent1[cross_i:]
        return child1, child2

    def run(self, iterations):
        # Initialize new population
        self._initialize()

        for epoch in range(iterations):
            population_fitness = self._determine_fitness()

            fittest_individual = self.population[np.argmax(population_fitness)]
            highest_fitness = max(population_fitness)

            # If we have found individual which matches the target => Done
            if fittest_individual == self.target:
                break

            # Calculate the probabilities that the individuals should be selected as
            # parents as the individuals fitness divided by the total population fitness
            parent_probs = [fitness / sum(population_fitness) for fitness in population_fitness]

            # Determine the next generation
            new_population = []
            for i in np.arange(0, self.population_size, 2):
                # Select two parents randomly according to probabilities
                parents = np.random.choice(self.population, size=2, p=parent_probs, replace=False)
                # Perform crossover to produce offspring
                child1, child2 = self._crossover(parents[0], parents[1])
                # Save mutated offspring for next generation
                new_population += [self._mutate(child1), self._mutate(child2)]

            print ("[%d Best Candidate: %s, Fitness: %.2f]" % (epoch, fittest_individual, highest_fitness))
            self.population = new_population

        print ("Answer: %s" % fittest_individual)


def main():
    target_string = "Genetic Algorithm"
    population_size = 100
    mutation_rate = 0.05
    genetic_algorithm = GeneticAlgorithm(target_string,
                                        population_size,
                                        mutation_rate)

    print ("")
    print ("+--------+")
    print ("|   GA   |")
    print ("+--------+")
    print ("Target String: %s" % target_string)
    print ("Population Size: %d" % population_size)
    print ("Mutation Rate: %s" % mutation_rate)
    print ("")

    genetic_algorithm.run(iterations=1000)

if __name__ == "__main__":
    main()






