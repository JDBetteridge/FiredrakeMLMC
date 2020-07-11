import numpy as np
from randomgen import RandomGenerator, MT19937

from MLMCv4 import MLMC_general_scalar, MLMC_Solver, P_level, convergence_tests

class binProblem:
    def __init__(self, level):
        self._bins = 30*(2**(level))

    def solve(self,sample):
        solution = np.histogram(sample, self._bins, (0,1000))
        ans = sum(i>0 for i in solution[0])
        #print(ans)
        return ans

rg = RandomGenerator(MT19937(12345))
def samp(level):
    ans = [rg.random_sample()*1000 for i in range(10)]
    return ans, ans


def general_test():
    # Levels and repetitions
    levels = 4
    repetitions = [1000, 100, 50, 10]
    print(MLMC_general_scalar(binProblem, samp, levels, repetitions, False))


if __name__ == '__main__':
    general_test()
    
    