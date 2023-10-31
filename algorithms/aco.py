from utils import BinaryKnapsackProblem, LoggerCSV
from numba.typed import List
import numpy as np
from utils import solutions_same


def run_aco(problem: BinaryKnapsackProblem, config, logger: LoggerCSV):
    ants_population = init_ants(problem, config.aco.population_size)
    item_lst = init_items(problem)

    while config.aco.generations != logger.cycle_count and \
            not solutions_same(logger.solution_of_best[len(logger.solution_of_best)-1], problem.solution):
        reset_all_ants(ants_population)
        
        ants_walk(ants_population, item_lst, problem.capacity)

        # print([item.pheromones for item in item_lst])
        update_pheromones(ants_population, item_lst,
                          problem.profits, config.aco.rho)
        # print([item.pheromones for item in item_lst])

        print([ant.genome for ant in ants_population])

        update_fitness(ants_population)
        logger.update_cycle(ants_population)
        logger.print_cycle_stats()


class Item:
    def __init__(self, value: int, weight: int, n_items: int, id: int) -> None:
        self.value = value
        self.weight = weight

        self.attractiveness = value / (weight)**2

        self.pheromones = np.ones(n_items)

        self.id = id


class Ant:
    def __init__(self, problem_size: int) -> None:
        self.genome = np.zeros(problem_size, dtype=int)
        self.items_picked: list[Item] = []
        self.fitness = 0

    def reset_solution(self):
        self.genome = np.zeros(len(self.genome), dtype=int)
        self.items_picked = []

    def get_carry_weight(self):
        s = 0

        for item in self.items_picked:
            s += item.weight

        return s

    def get_carry_profit(self):
        s = 0

        for item in self.items_picked:
            s += item.value

        return s

    def pick_up_item(self, idx: int, item: Item):
        self.genome[idx] += 1
        self.items_picked.append(item)


def update_fitness(ants_population: list[Ant]):
    for ant in ants_population:
        ant.fitness = ant.get_carry_profit()

def reset_all_ants(ants_population: list[Ant]):
    for ant in ants_population:
        ant.reset_solution()


def init_items(problem: BinaryKnapsackProblem) -> list[Item]:

    item_lst = []  # List()

    for i in range(len(problem.profits)):
        item = Item(problem.profits[i],
                    problem.weights[i], len(problem.profits), i)
        item_lst.append(item)

    return item_lst


def init_ants(problem: BinaryKnapsackProblem, population_size: int):
    ants_population = []  # List()

    for i in range(population_size):
        ant = Ant(len(problem.profits))
        ants_population.append(ant)

    return ants_population


def ants_walk(ants_population: list[Ant], item_lst: list[Item], capacity: int):
    for ant in ants_population:
        walk(ant, item_lst, capacity)


def walk(ant: Ant, item_lst: list[Item], capacity: int):
    ant_capacity = ant.get_carry_weight()
    pool_idx = create_item_pool(ant, ant_capacity, item_lst)
    # print(ant_capacity)
    # print(ant.genome)
    # print('pool of indexes to pick from', pool_idx)

    last_item = None

    while len(pool_idx) > 0:

        pool_idx_probs = selection_probabilities(pool_idx, item_lst, last_item)
        # print('pool of indexes to pick from', pool_idx)

        selected_idx = np.random.choice(pool_idx, 1, p=pool_idx_probs)

        ant.pick_up_item(selected_idx[0], item_lst[selected_idx[0]])
        last_item = ant.items_picked[-1]

        # print()
        # print(pool_idx)
        # print([item_lst[i].attractiveness for i in pool_idx])
        # print([item_lst[i].pheromones[i] for i in pool_idx])
        # print(pool_idx_probs)
        # print()

        ant_capacity = ant.get_carry_weight()
        pool_idx = create_item_pool(ant, ant_capacity, item_lst)


def create_item_pool(ant: Ant, ant_capacity: int, item_lst: list[Item]):
    pool_idx = []

    for i in range(len(item_lst)):
        if (item_lst[i] not in ant.items_picked) and (item_lst[i].weight > ant_capacity):
            pool_idx.append(i)

    return pool_idx


def selection_probabilities(pool_idx: list[int], item_lst: list[Item], last_item: Item = None):
    probabilities = []
    sum_attract_phero = sum_for_selection_probabilities(
        pool_idx, item_lst, last_item)

    for idx in pool_idx:
        if last_item is None:
            p = (item_lst[idx].attractiveness * item_lst[idx].pheromones[idx]) / sum_attract_phero
        else:
            p = (item_lst[idx].attractiveness * last_item.pheromones[idx]) / sum_attract_phero

        probabilities.append(p)

    return probabilities


def sum_for_selection_probabilities(pool_idx: list[int], item_lst: list[Item], last_item: Item = None):
    s = 0

    for idx in pool_idx:
        if last_item is None:
            s += item_lst[idx].attractiveness*item_lst[idx].pheromones[idx]
        else:
            s += item_lst[idx].attractiveness*last_item.pheromones[idx]

    return s


def update_pheromones(ants_population: list[Ant], item_lst: list[Item], profits: list[int], evaporation_rate: int):
    ants_value = []

    for ant in ants_population:
        value = np.sum(ant.genome * profits)
        ants_value.append(value)

    best_value = np.max(ants_value)
    best_value_idx = np.argmax(ants_value)

    for i in range(len(ants_population)):
        update_from_solution(
            ants_population[i], item_lst, ants_value[i], best_value)

    evaporate_pheromones(item_lst, evaporation_rate)


def update_from_solution(ant: Ant, item_lst: list[Item], ant_value: int, best_value: int):
    prev_idx = None

    for i in range(len(ant.items_picked)):
        if prev_idx is None:
            item_idx = item_lst.index(ant.items_picked[i])
            print('best_value:', best_value, 'ant_value:', ant_value)
            print('pheromone placed', 1 / (1 + ((best_value - ant_value) / best_value)))
            ant.items_picked[i].pheromones[item_idx] += 1 / (1 + ((best_value - ant_value) / best_value))
            prev_idx = item_idx
        else:
            item_idx = item_lst.index(ant.items_picked[i])
            ant.items_picked[i].pheromones[prev_idx] += 1 / (1 + ((best_value - ant_value) / best_value))
            ant.items_picked[i-1].pheromones[item_idx] += 1 / (1 + ((best_value - ant_value) / best_value))
            prev_idx = item_idx


def evaporate_pheromones(item_lst: list[Item], evaporation_rate: int):
    for item in item_lst:
        for i in range(len(item.pheromones)):
            item.pheromones[i] *= evaporation_rate
