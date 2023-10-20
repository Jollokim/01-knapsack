import numpy as np
from numba.experimental import jitclass
from numba import int32, string



class BinaryKnapsackProblem:
    def __init__(self, name: str, problem_dir: str) -> None:
        self.name = name
        self.problem_dir = problem_dir

        self.load_problem()


    def load_problem(self):
        self.capacity = self._load_capacity()

        self.profits = self._load_array('p')
        self.weights = self._load_array('w')
        self.solution = self._load_array('s')

    def _load_capacity(self):
        capacity_file = f'{self.problem_dir}/{self.name}_c.txt'

        with open(capacity_file, 'r') as file:
            num = file.readline()
            num = int(num)

        return num
    
    def _load_array(self, attribute: str):
        arr_file = f'{self.problem_dir}/{self.name}_{attribute}.txt'

        
        with open(arr_file, 'r') as file:
            lines = file.readlines()
            arr = np.zeros(len(lines), dtype=int)

            for i in range(len(lines)):
                arr[i] = int(lines[i])


        return arr







if __name__ == '__main__':
    problem = BinaryKnapsackProblem('p01', 'data\p01')

