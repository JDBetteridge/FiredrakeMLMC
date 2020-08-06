import numpy as np
from randomgen import RandomGenerator, MT19937

from MLMCv5 import MLMC_Solver, MLMC_Problem, do_MC

class binProblem:
    def __init__(self, level_obj):
        self._bins = 10*(5**(level_obj))

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
    print(ans)
    if level2 == None:
        return ans, None
    return ans, ans


def general_test():
    # Levels and repetitions
    levels = 3
    repetitions = [10, 5, 2]
    MLMCprob = MLMC_Problem(binProblem, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate = MLMCsolv.solve()
    print(estimate)

def manual_test(samples):
    # made for 17 samples
    level0 = binProblem(0)
    level1 = binProblem(1)
    level2 = binProblem(2)
    level0_results = [level0.solve(samples[i]) for i in range(10)]
    level1_results = [[level1.solve(samples[i]), level0.solve(samples[i])] for i in range (10, 15)] 
    level2_results = [[level2.solve(samples[i]), level1.solve(samples[i])] for i in range (15, 17)]

    L0 = sum(level0_results)/len(level0_results)
    L1_sub = [i[0]-i[1] for i in level1_results]
    L1 = sum(L1_sub)/len(L1_sub)
    L2_sub = [i[0]-i[1] for i in level2_results]
    L2 = sum(L2_sub)/len(L2_sub)

    print(L0+L1+L2)
    #result is 10.8




if __name__ == '__main__':
    general_test()
    #[rg.random_sample()*1000 for i in range(5)]
    #rg = RandomGenerator(MT19937(12345))
    #ans = [[rg.random_sample()*1000 for i in range(10)] for i2 in range(17)]
    #manual_test(ans)
    
    