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

        self.solution = problem.solution

        self.cycle_count = 0
        
        self.best_value = 0
        self.best_value_cycle = 0
        self.best_solution = []
        self.best_closeness = len(self.solution)


    def get_solution_closeness(self, solution: np.ndarray):
        count = 0
        for i in range(len(self.solution)):
            if solution[i] == self.solution[i]:
                count += 1

        closeness = len(solution) - count

        return closeness
    
    # NOTE: make common class for individual and Ants
    def update_cycle(self, population: list):
        population_fitnesses = [individual.fitness for individual in population]

        best = np.max(population_fitnesses)
        avg = np.mean(population_fitnesses)

        self.best_value_of_cycle.append(best)
        self.avg_value_of_cycle.append(avg)

        best_idx = np.argmax(population_fitnesses)

        self.solution_of_best.append(population[best_idx].genome)

        if best > self.best_value:
            self.best_value = best
            self.best_value_cycle = self.cycle_count
            self.best_solution = population[best_idx].genome
            self.best_closeness = self.get_solution_closeness(self.best_solution)

        self.cycle_count += 1
    
    def print_cycle_stats(self):
        print('Cycle:', len(self.best_value_of_cycle)-1)
        print('Best value:', self.best_value_of_cycle[len(self.best_value_of_cycle)-1])
        print('Mean value:', self.avg_value_of_cycle[len(self.best_value_of_cycle)-1])

    def write_csv(self, path: str):
        df = pd.DataFrame(
            {
                'best': self.best_value_of_cycle,
                'avg': self.avg_value_of_cycle
            }
        )

        df.to_csv(f'{path}', index=True)

    def write_best_stats(self, path: str):
        with open(path, 'w') as file:
            file.write('cycle,best_value,solution,closeness\n')
            file.write(f'{self.best_value},{self.best_value_cycle},{self.best_solution},{self.best_closeness}')