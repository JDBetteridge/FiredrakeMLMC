import numpy as np
from randomgen import RandomGenerator, MT19937

from MLMCv4 import MLMC_general_scalar, MLMC_Solver, P_level, convergence_tests

class binProblem:
    def __init__(self, level):
        self._bins = 30*(2**(level))

    def solve(self,sample):
        solution = np.histogram(sample, self._bins, (0,1000))
        ans = sum(i>0 for i in solution[0])
        return ans

rg = RandomGenerator(MT19937(12345))
def samp():
    return [rg.random_sample()*1000 for i in range(10)]


def general_test():
    # Levels and repititions
    levels = 3
    repititions = [1000, 100, 10]
    print(MLMC_general_scalar(binProblem, samp, levels, repititions, False))


if __name__ == '__main__':
    general_test()
    
    