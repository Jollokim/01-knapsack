from utils import BinaryKnapsackProblem, LoggerCSV
from numba.typed import List
import numpy as np


def run_aco(problem: BinaryKnapsackProblem, config, logger: LoggerCSV):
    ants_population = init_ants(problem, config.aco.population_size)
    item_lst = init_items(problem)
    

    for cycle in range(config.aco.generations):
        ants_walk(ants_population, item_lst, problem.capacity)
        update_pheromones(ants_population, config.aco.rho)

        print([ant.solution for ant in ants_population])
        quit()

        


class Item:
    def __init__(self, value: int, weight: int, n_items: int, id: int) -> None:
        self.value = value
        self.weight = weight

        self.attractiveness = value / (weight)**2

        self.pheromones = np.ones(n_items)

        self.id = id


class Ant:
    def __init__(self, problem_size: int) -> None:
        self.solution = np.zeros(problem_size, dtype=int)
        self.items_picked: list[Item] = []

    def reset_solution(self):
        self.solution = np.zeros(len(self.solution), dtype=int)

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
        self.solution[idx] += 1
        self.items_picked.append(item)


def init_items(problem: BinaryKnapsackProblem):

    item_lst = [] # List()

    for i in range(len(problem.profits)):
        item = Item(problem.profits[i],
                    problem.weights[i], len(problem.profits), i)
        item_lst.append(item)

    return item_lst


def init_ants(problem: BinaryKnapsackProblem, population_size: int):
    ants_population = [] # List()

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

    last_item = None

    while len(pool_idx) > 0:

        pool_idx_probs = selection_probabilities(pool_idx, item_lst, last_item)

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



def selection_probabilities(pool_idx: list[int], item_lst: list[Item], last_item: Item=None):
    probabilities = []
    sum_attract_phero = sum_for_selection_probabilities(pool_idx, item_lst)

    for idx in pool_idx:
        if last_item is None:
            p = item_lst[idx].attractiveness*item_lst[idx].pheromones[idx] / sum_attract_phero
        else:
            p = item_lst[idx].attractiveness*last_item.pheromones[idx] / sum_attract_phero

        probabilities.append(p)

    return probabilities



def sum_for_selection_probabilities(pool_idx: list[int], item_lst: list[Item], last_item: Item=None):
    s = 0

    for idx in pool_idx:
        if last_item is None:
            s += item_lst[idx].attractiveness*item_lst[idx].pheromones[idx]
        else:
            s += item_lst[idx].attractiveness*last_item.pheromones[idx]

    return s


def update_pheromones(ants_population: list[Ant], evaporation_rate: int):
    pass