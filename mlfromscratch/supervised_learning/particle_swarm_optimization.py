from __future__ import print_function, division
import numpy as np
import copy

class ParticleSwarmOptimizedNN():
    """ Particle Swarm Optimization of Neural Network.

    Parameters:
    -----------
    n_individuals: int
        The number of neural networks that are allowed in the population at a time.
    model_builder: method
        A method which returns a user specified NeuralNetwork instance.
    inertia_weight:     float [0,1)
    cognitive_weight:   float [0,1)
    social_weight:      float [0,1)
    max_velocity: float
        The maximum allowed value for the velocity.

    Reference:
        Neural Network Training Using Particle Swarm Optimization
        https://visualstudiomagazine.com/articles/2013/12/01/neural-network-training-using-particle-swarm-optimization.aspx 
    """
    def __init__(self, population_size, 
                        model_builder, 
                        inertia_weight=0.8, 
                        cognitive_weight=2, 
                        social_weight=2, 
                        max_velocity=20):
        self.population_size = population_size
        self.model_builder = model_builder
        self.best_individual = None
        # Parameters used to update velocity
        self.cognitive_w = cognitive_weight
        self.inertia_w = inertia_weight
        self.social_w = social_weight
        self.min_v = -max_velocity
        self.max_v = max_velocity

    def _build_model(self, id):
        """ Returns a new individual """
        model = self.model_builder(n_inputs=self.X.shape[1], n_outputs=self.y.shape[1])
        model.id = id
        model.fitness = 0
        model.highest_fitness = 0
        model.accuracy = 0
        # Set intial best as the current initialization
        model.best_layers = copy.copy(model.layers)

        # Set initial velocity to zero
        model.velocity = []
        for layer in model.layers:
            velocity = {"W": 0, "w0": 0}
            if hasattr(layer, 'W'):
                velocity = {"W": np.zeros_like(layer.W), "w0": np.zeros_like(layer.w0)}
            model.velocity.append(velocity)

        return model

    def _initialize_population(self):
        """ Initialization of the neural networks forming the population"""
        self.population = []
        for i in range(self.population_size):
            model = self._build_model(id=i)
            self.population.append(model)

    def _update_weights(self, individual):
        """ Calculate the new velocity and update weights for each layer """
        # Two random parameters used to update the velocity
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        for i, layer in enumerate(individual.layers):
            if hasattr(layer, 'W'):
                # Layer weights velocity
                first_term_W = self.inertia_w * individual.velocity[i]["W"]
                second_term_W = self.cognitive_w * r1 * (individual.best_layers[i].W - layer.W)
                third_term_W = self.social_w * r2 * (self.best_individual.layers[i].W - layer.W)
                new_velocity = first_term_W + second_term_W + third_term_W
                individual.velocity[i]["W"] = np.clip(new_velocity, self.min_v, self.max_v)

                # Bias weight velocity
                first_term_w0 = self.inertia_w * individual.velocity[i]["w0"]
                second_term_w0 = self.cognitive_w * r1 * (individual.best_layers[i].w0 - layer.w0)
                third_term_w0 = self.social_w * r2 * (self.best_individual.layers[i].w0 - layer.w0)
                new_velocity = first_term_w0 + second_term_w0 + third_term_w0
                individual.velocity[i]["w0"] = np.clip(new_velocity, self.min_v, self.max_v)

                # Update layer weights with velocity
                individual.layers[i].W += individual.velocity[i]["W"]
                individual.layers[i].w0 += individual.velocity[i]["w0"]
        
    def _calculate_fitness(self, individual):
        """ Evaluate the individual on the test set to get fitness scores """
        loss, acc = individual.test_on_batch(self.X, self.y)
        individual.fitness = 1 / (loss + 1e-8)
        individual.accuracy = acc

    def evolve(self, X, y, n_generations):
        """ Will evolve the population for n_generations based on dataset X and labels y"""
        self.X, self.y = X, y

        self._initialize_population()

        # The best individual of the population is initialized as population's first ind.
        self.best_individual = copy.copy(self.population[0])

        for epoch in range(n_generations):
            for individual in self.population:
                # Calculate new velocity and update the NN weights
                self._update_weights(individual)
                # Calculate the fitness of the updated individual
                self._calculate_fitness(individual)

                # If the current fitness is higher than the individual's previous highest
                # => update the individual's best layer setup
                if individual.fitness > individual.highest_fitness:
                    individual.best_layers = copy.copy(individual.layers)
                    individual.highest_fitness = individual.fitness
                # If the individual's fitness is higher than the highest recorded fitness for the
                # whole population => update the best individual
                if individual.fitness > self.best_individual.fitness:
                    self.best_individual = copy.copy(individual)

            print ("[%d Best Individual - ID: %d Fitness: %.5f, Accuracy: %.1f%%]" % (epoch,
                                                                            self.best_individual.id,
                                                                            self.best_individual.fitness,
                                                                            100*float(self.best_individual.accuracy)))
        return self.best_individual

