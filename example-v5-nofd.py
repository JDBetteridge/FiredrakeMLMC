import numpy as np
from randomgen import RandomGenerator, MT19937

from MLMCv5 import MLMC_Solver, MLMC_Problem, do_MC

class binProblem:
    def __init__(self, level_obj):
        self._bins = 30*(2**(level_obj))

    def solve(self, sample):
        solution = np.histogram(sample, self._bins, (0,1000))
        ans = sum(i>0 for i in solution[0])
        #print(ans)
        return ans

def lvl_maker(level_f, level_c):
    if level_c < 0:
        return level_f, None
    else:
        return level_f, level_c

rg = RandomGenerator(MT19937(12345))
def samp(level, level2):
    ans = [rg.random_sample()*1000 for i in range(10)]
    if level2 == None:
        return ans, None
    return ans, ans


def general_test():
    # Levels and repetitions
    levels = 4
    repetitions = [100, 100, 100, 100]
    MLMCprob = MLMC_Problem(binProblem, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate = MLMCsolv.solve()
    print(estimate)


if __name__ == '__main__':
    general_test()
    #[rg.random_sample()*1000 for i in range(5)]
    
    