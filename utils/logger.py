import numpy as np
import pandas as pd
from utils import BinaryKnapsackProblem


class LoggerCSV():
    def __init__(self, problem: BinaryKnapsackProblem) -> None:
        self.problem: BinaryKnapsackProblem = problem

        self.best_value_of_cycle = [0]
        self.avg_value_of_cycle = [0]

        self.weight_of_best = [0]
        self.solution_of_best = [np.zeros(len(problem.profits))]

        self.cycle_count = 0
    
    # NOTE: make common class for individual and Ants
    def update_cycle(self, population: list):
        population_fitnesses = [individual.fitness for individual in population]

        best = np.max(population_fitnesses)
        avg = np.mean(population_fitnesses)

        self.best_value_of_cycle.append(best)
        self.avg_value_of_cycle.append(avg)

        best_idx = np.argmax(population_fitnesses)

        self.solution_of_best.append(population[best_idx].genome)

        self.cycle_count += 1
    
    def print_cycle_stats(self):
        print('Cycle:', len(self.best_value_of_cycle)-1)
        print('Best value:', self.best_value_of_cycle[len(self.best_value_of_cycle)-1])
        print('Mean value:', self.avg_value_of_cycle[len(self.best_value_of_cycle)-1])

    def write_csv(self, name: str):
        df = pd.DataFrame(
            {
                'best': self.best_value_of_cycle,
                'avg': self.avg_value_of_cycle
            }
        )

        df.to_csv(f'{name}.csv')