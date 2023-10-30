from utils import BinaryKnapsackProblem, LoggerCSV
from algorithms import run_ga
from omegaconf import OmegaConf
import os


class DummyConfig:
    def __init__(self) -> None:
        self.population_size = 10
        self.mutation_rate = 0.5

        self.n_cycles = 100

def main():
    problem_name = 'p01' 
    problem_dir = 'data\p01'
    config = OmegaConf.load('configs\ga.yaml')

    problem = BinaryKnapsackProblem(config.problem.problem_name, config.problem.problem_dir)
    print('Weights:')
    print(problem.weights)
    print('Profits:')
    print(problem.profits)
    print('Solution:')
    print(problem.solution)
    print('Optimal profit:')
    print(problem.optimal_profit)
    
    logger = LoggerCSV(problem)

    results_dir = f'results/{config.problem.problem_name}_{config.problem.algorithm}'
    os.makedirs(results_dir, exist_ok=True)

    run_ga(problem, config, logger)

    logger.write_csv(f'{results_dir}/result.csv')







if __name__ == '__main__':
    main()