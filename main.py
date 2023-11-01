from utils import BinaryKnapsackProblem, LoggerCSV, plot_run
from algorithms import run_ga, run_aco
from omegaconf import OmegaConf
import os
import argparse


algorithm_function = {'ga': run_ga, 'aco': run_aco}


def main(args):
    config = OmegaConf.load(args.config_file)

    problem = BinaryKnapsackProblem(
        config.problem.problem_name, config.problem.problem_dir)
    print('Capacity:')
    print(problem.capacity)
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

    run_func = algorithm_function[config.problem.algorithm]

    run_func(problem, config, logger)

    print(problem.solution)

    logger.write_csv(f'{results_dir}/result.csv')
    logger.write_best_stats(f'{results_dir}/best.txt')

    plot_run(logger.best_value_of_cycle[1:], 'best value of cycle',
             f'{results_dir}/plots/best_value_of_cycle.png')
    plot_run(logger.avg_value_of_cycle[1:], 'average value of cycle',
             f'{results_dir}/plots/avg_value_of_cycle.png')


def args_parser():
    parser = argparse.ArgumentParser(
        prog='Evolutionary algorithms for binary knapsack problem',
        description='Uses ga or aco to find best solution for binary knapsack problem'
                                     )
    
    parser.add_argument('--config_file', type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parser()
    main(args)
