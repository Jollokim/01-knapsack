import numpy as np

from utils import BinaryKnapsackProblem, solutions_same
from utils import LoggerCSV


def run_ga(problem: BinaryKnapsackProblem, config, logger: LoggerCSV):
    item_profits = problem.profits
    item_weights = problem.weights
    restraint = problem.capacity

    population = init_population(config.ga.population_size, len(item_profits))

    # unflip so to not exceed the restraint
    unflip_to_fit_restraint(population, item_weights, restraint)
    # update fitness on all individuals
    update_population_fitness(population, item_profits)
    
    while config.ga.generations != logger.cycle_count and \
        not solutions_same(logger.solution_of_best[len(logger.solution_of_best)-1], problem.solution):

        logger.start_cycle_timing()
        # select parent
        parents = select_parents_roulette_wheel(population, elites=config.ga.elites)
        # print([parent.genome for parent in parents])

        # crossover to create children
        children = crossover_parents(parents)

        # mutate children
        mutate_population(children, config.ga.mutation_rate)

        # gather entire population of parents and children
        population = concat_parents_children(parents, children)

        # unflip so to not exceed the restraint
        unflip_to_fit_restraint(population, item_weights, restraint)

        # update fitness on all individuals
        update_population_fitness(population, item_profits)

        logger.end_cycle_timing()
        logger.update_cycle(population)
        logger.print_cycle_stats()



    print([population[i].fitness for i in range(len(population))])
    print([individual.genome for individual in population])



class Individual():
    def __init__(self, geno_size: int) -> None:
        self.genome = np.zeros(geno_size, dtype=np.int32)
        self.fitness = 0

    def init_random(self):
        for i in range(len(self.genome)):
            self.genome[i] = np.floor(np.random.random() * 2)

    def update_fitness(self, fitness: int):
        self.fitness = fitness

    def __str__(self) -> str:
        return f'{self.genome}, fitness:{self.fitness}'


def init_population(population_size: int, geno_size: int) -> list[Individual]:
    population = []

    for i in range(population_size):
        individual = Individual(geno_size)
        individual.init_random()

        population.append(individual)

    return population


def fitness_f(individual: Individual, item_profits: np.array) -> int:
    fitness = np.sum(individual.genome * item_profits)
    return fitness


def weight_f(individual: Individual, item_weights: np.array) -> int:
    weight = fitness_f(individual, item_weights)
    return weight


def update_population_fitness(population: list[Individual], item_profits: np.array):
    for i in range(len(population)):
        fitness = fitness_f(population[i], item_profits)

        population[i].update_fitness(fitness)


def total_population_fitness(population: list[Individual]) -> int:
    total_fitness = 0

    for individual in population:
        total_fitness += individual.fitness

    return total_fitness


def roulette_wheel_probabilities(population: list[Individual], population_fitness: int) -> list[float]:
    probabilities = []

    for individual in population:
        probability = individual.fitness / population_fitness
        probabilities.append(probability)

    return probabilities


def select_parents_roulette_wheel(population: list[Individual], elites: int=0) -> list[Individual]:
    population_fitness = total_population_fitness(population)

    probabilities = roulette_wheel_probabilities(
        population, population_fitness)

    sel_parents = np.random.choice(
        population, size=(len(population)//2)-elites, p=probabilities)
    
    best_fitness_idx = np.argmax([individual.fitness for individual in population])

    parents = np.array([None for i in range(len(population)//2)], dtype=Individual)

    parents[0] = population[best_fitness_idx]
    parents[1:] = sel_parents
    


    return parents


def random_int_from_to(min: int, max: int):
    # drawing a random point in interval [min , max) and flooring it to get index
    point = int(np.floor((np.random.random() * (max-min)) + min))

    return point


def single_point_crossover(parent1: Individual, parent2: Individual) -> list[Individual]:

    point = random_int_from_to(1, len(parent1.genome)-1)

    genome_p1_1 = parent1.genome[:point]
    genome_p1_2 = parent1.genome[point:]

    genome_p2_1 = parent2.genome[:point]
    genome_p2_2 = parent2.genome[point:]

    child1 = Individual(len(parent1.genome))
    child2 = Individual(len(parent1.genome))

    child1.genome[:point] = genome_p1_1
    child1.genome[point:] = genome_p2_2

    child2.genome[:point] = genome_p2_1
    child2.genome[point:] = genome_p1_2

    children = [child1, child2]

    return children


def crossover_parents(parents: list[Individual]) -> list[Individual]:
    children = []

    # create pairs
    for i in range(1, len(parents), 2):
        pair = [parents[i-1], parents[i]]
        pairs_children = single_point_crossover(pair[0], pair[1])

        children.append(pairs_children[0])
        children.append(pairs_children[1])

    return children


def concat_parents_children(parents: list[Individual], children: list[Individual]) -> list[Individual]:
    population = []

    for i in range(len(parents)):
        population.append(parents[i])

    for i in range(len(children)):
        population.append(children[i])

    return population


def mutate_population(population: list[Individual], mutation_rate: int = 0.5):
    for individual in population:
        for i in range(len(individual.genome)):
            flip = np.random.random()

            if flip < 0.5:
                if individual.genome[i] == 0:
                    individual.genome[i] = 1
                else:
                    individual.genome[i] = 0
            else:
                continue


def unflip_to_fit_restraint(population: list[Individual], item_weights: np.array, restraint: int):
    for individual in population:
        ones_idx = []

        for i in range(len(individual.genome)):
            if individual.genome[i] == 1:
                ones_idx.append(i)

        while weight_f(individual, item_weights) > restraint:
            random_index = random_int_from_to(0, len(ones_idx))

            individual.genome[ones_idx[random_index]] = 0

            del ones_idx[random_index]

