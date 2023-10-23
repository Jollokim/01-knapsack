import numpy as np
from numba import njit, int32
from numba.experimental import jitclass

from numba.typed import List
from utils import BinaryKnapsackProblem
from utils import LoggerCSV


def run_ga(problem: BinaryKnapsackProblem, config, logger: LoggerCSV):
    item_profits = problem.profits
    item_weights = problem.weights
    restraint = problem.capacity

    population = init_population(config.population_size, len(item_profits))

    while True:
        # unflip so to not exceed the restraint
        unflip_to_fit_restraint(population, item_weights, restraint)

        # update fitness on all individuals
        update_population_fitness(population, item_profits)

        logger.update_cycle(population)
        logger.print_cycle_stats()

        print(np.max([individual.fitness for individual in population]))

        # select parent
        parents = select_parents_roulette_wheel(population)
        # print([parent.genome for parent in parents])

        # crossover to create children
        children = crossover_parents(parents)

        # mutate children
        mutate_population(children, config.mutation_rate)

        # gather entire population of parents and children
        population = concat_parents_children(parents, children)

    print([population[i].fitness for i in range(len(population))])
    print([individual.genome for individual in population])


specs = [
    ('genome', int32[:]),
    ('fitness', int32)
]


@jitclass(specs)
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


@njit
def init_population(population_size: int, geno_size: int) -> list[Individual]:
    population = List()

    for i in range(population_size):
        individual = Individual(geno_size)
        individual.init_random()

        population.append(individual)

    return population


@njit
def fitness_f(individual: Individual, item_profits: np.array) -> int:
    fitness = np.sum(individual.genome * item_profits)
    return fitness


@njit
def weight_f(individual: Individual, item_weights: np.array) -> int:
    weight = fitness_f(individual, item_weights)
    return weight


@njit
def update_population_fitness(population: list[Individual], item_profits: np.array):
    for i in range(len(population)):
        fitness = fitness_f(population[i], item_profits)

        population[i].update_fitness(fitness)


@njit
def total_population_fitness(population: list[Individual]) -> int:
    total_fitness = 0

    for individual in population:
        total_fitness += individual.fitness

    return total_fitness


@njit
def roulette_wheel_probabilities(population: list[Individual], population_fitness: int) -> list[float]:
    probabilities = []

    for individual in population:
        probability = individual.fitness / population_fitness
        probabilities.append(probability)

    return probabilities


# @njit
def select_parents_roulette_wheel(population: list[Individual]) -> list[Individual]:
    population_fitness = total_population_fitness(population)

    probabilities = roulette_wheel_probabilities(
        population, population_fitness)

    parents = np.random.choice(
        population, size=len(population)//2, p=probabilities)

    return parents


@njit
def random_int_from_to(min: int, max: int):
    # drawing a random point in interval [min , max) and flooring it to get index
    point = int(np.floor((np.random.random() * (max-min)) + min))

    return point


@njit
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

    children = List([child1, child2])

    return children


# @njit
def crossover_parents(parents: list[Individual]) -> list[Individual]:
    children = List()

    # create pairs
    for i in range(1, len(parents), 2):
        pair = [parents[i-1], parents[i]]
        pairs_children = single_point_crossover(pair[0], pair[1])

        children.append(pairs_children[0])
        children.append(pairs_children[1])

    return children


# @njit
def concat_parents_children(parents: list[Individual], children: list[Individual]) -> list[Individual]:
    population = List()

    for i in range(len(parents)):
        population.append(parents[i])

    for i in range(len(children)):
        population.append(children[i])

    return population


@njit
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


@njit
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


# if __name__ == '__main__':
#     run_ga()