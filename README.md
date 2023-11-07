# 01-knapsack
Solving 0/1 knapsack with genetic algorithm and ant colony optimization

## Get started.

Install Python3.11 or higher. Older versions of Python might also work but make sure atleast to have Python3.

## Preparing the python environment
### Create the python environment:

    python3 -m venv <environment_path>

### Activate the environment (Linux)

    source <environment_path>/bin/activate

### Installing dependecies
Please activate your environment before doing the next step. The environment activation method is different based on your operating system.

    pip install -r requirements.txt

## How to:

    python main.py --config_file <path_to_yaml_config>

Example:
    
    python main.py --config_file configs\aco_r30.yaml