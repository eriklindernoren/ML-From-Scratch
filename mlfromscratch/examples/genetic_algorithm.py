
from mlfromscratch.unsupervised_learning import GeneticAlgorithm

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
    print ("Description: Implementation of a Genetic Algorithm which aims to produce")
    print ("the user specified target string. This implementation calculates each")
    print ("candidate's fitness based on the alphabetical distance between the candidate")
    print ("and the target. A candidate is selected as a parent with probabilities proportional")
    print ("to the candidate's fitness. Reproduction is implemented as a single-point")
    print ("crossover between pairs of parents. Mutation is done by randomly assigning")
    print ("new characters with uniform probability.")
    print ("")
    print ("Parameters")
    print ("----------")
    print ("Target String: '%s'" % target_string)
    print ("Population Size: %d" % population_size)
    print ("Mutation Rate: %s" % mutation_rate)
    print ("")

    genetic_algorithm.run(iterations=1000)

if __name__ == "__main__":
    main()