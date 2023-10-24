from utils import BinaryKnapsackProblem, LoggerCSV
from algorithms import run_ga
from omegaconf import OmegaConf


class DummyConfig:
    def __init__(self) -> None:
        self.population_size = 10
        self.mutation_rate = 0.5

        self.n_cycles = 100

def main():
    problem_name = 'p08' 
    problem_dir = 'data\p08'

    problem = BinaryKnapsackProblem(problem_name, problem_dir)
    print('Weights:')
    print(problem.weights)
    print('Profits:')
    print(problem.profits)
    print('Solution:')
    print(problem.solution)
    print('Optimal profit:')
    print(problem.optimal_profit)
    
    logger = LoggerCSV(problem)

    config = OmegaConf.load('configs\ga.yaml')
    
    run_ga(problem, config, logger)







if __name__ == '__main__':
    main()