import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from randomgen import RandomGenerator, MT19937

from mlmcparagen import MLMC_Solver, MLMC_Problem, do_MC

class binProblem:
    def __init__(self, level_obj):
        self._bins = 10*(5**(level_obj))

    def solve(self, sample):
        solution = np.histogram(sample, self._bins, (0,1000))
        ans = sum(i>0 for i in solution[0])
        return ans

def lvl_maker(level_f, level_c, comm = MPI.COMM_WORLD):
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
    repetitions = [1000, 500, 1000, 1000]
    MLMCprob = MLMC_Problem(binProblem, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate = MLMCsolv.solve()
    print(estimate)

def manual_test(samples):
    # made for 17 samples
    level0 = binProblem(0)
    level1 = binProblem(1)
    level2 = binProblem(2)
    level0_results = [level0.solve(samples[i]) for i in range(100)]
    level1_results = [[level1.solve(samples[i]), level0.solve(samples[i])] for i in range (100, 150)]
    level2_results = [[level2.solve(samples[i]), level1.solve(samples[i])] for i in range (150, 160)]

    L0 = sum(level0_results)/len(level0_results)
    L1_sub = [i[0]-i[1] for i in level1_results]
    L1 = sum(L1_sub)/len(L1_sub)
    L2_sub = [i[0]-i[1] for i in level2_results]
    L2 = sum(L2_sub)/len(L2_sub)

    print(L0+L1+L2)
    #result is 10.8

def MC(reps, level):
    string = "BallDrop_{}r_{}lvl".format(reps, level)

    results = do_MC(binProblem, reps, level, samp)
    #with open(string+'.json', 'w') as f:
        #json.dump(results, f)

    res2 = [sum(results[:i+1])/(i+1) for i in range(len(results))]
    #fig, axes = plt.subplots()
    #axes.plot([i for i in range(reps)], res2, 'r')
    #plt.show()
    return res2[-1]

def test_mc():
    result = MC(1000, 3)
    assert (np.abs(result - 10) < 0.1 ), "MC failed to converge"

if __name__ == '__main__':
    #general_test()
    test_MC(1000, 3)
    #[rg.random_sample()*1000 for i in range(5)]
    #rg = RandomGenerator(MT19937(12345))
    #ans = [[rg.random_sample()*1000 for i in range(10)] for i2 in range(160)]
    #manual_test(ans)

