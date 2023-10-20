import numpy as np
from numba import njit, int32
from numba.experimental import jitclass

from numba.typed import List


def start_ga(args):
    pass


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
def init_population(size: int, geno_size: int) -> list[Individual]:
    population = List()

    for i in range(size):
        individual = Individual(geno_size)
        individual.init_random()

        population.append(individual)

    return population


@njit
def fitness_f(individual: Individual, item_profits: np.array) -> int:
    fitness = np.sum(individual.genome * item_profits)
    return fitness


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
    
    parents = np.random.choice(population, size=len(population)//2, p=probabilities)

    return parents


@njit
def single_point_crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    
    # drawing a random point in interval [1 , 10) and flooring it to get index
    min = 1
    max = len(parent1.genome)-1
    point = np.floor((np.random.random() * (max-min)) + min)




@njit
def crossover_parents(parents: list[Individual]) :
    children = []

    # create pairs 
    for i in range(1, len(parents), 2):
        pair = (parents[i-1], parents[i])
        pairs_children = single_point_crossover(pair[0], pair[1])

        children.append(pairs_children[0])
        children.append(pairs_children[1])

    children

    

    

        




def main():
    item_profits = np.array([10 for i in range(10)], dtype=np.int32)
    population = init_population(8, 10)

    print([population[i].fitness for i in range(len(population))])
    update_population_fitness(population, item_profits)
    print([population[i].fitness for i in range(len(population))])

    parents = select_parents_roulette_wheel(population)
    print(parents)


if __name__ == '__main__':
    main()
