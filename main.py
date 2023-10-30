from utils import BinaryKnapsackProblem, LoggerCSV, plot_run
from algorithms import run_ga
from omegaconf import OmegaConf
import os


def main():
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
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    run_ga(problem, config, logger)

    logger.write_csv(f'{results_dir}/result.csv')
    logger.write_best_stats(f'{results_dir}/best.txt')

    plot_run(logger.best_value_of_cycle[1:], 'best value of cycle', f'{results_dir}/plots/best_value_of_cycle.png')
    plot_run(logger.avg_value_of_cycle[1:], 'average value of cycle', f'{results_dir}/plots/avg_value_of_cycle.png')









if __name__ == '__main__':
    main()