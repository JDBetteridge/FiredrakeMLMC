import numpy as np
from mpi4py import MPI
from numpy.random import default_rng
from mlmcparagen import MLMC_Solver, MLMC_Problem

rng = default_rng()

class SimpleProblem(object):
    def __init__(self, grid):
        self.grid = grid
        boundaries = np.zeros(grid.size + 1)
        boundaries[1:-1] = (grid[:-1] + grid[1:])/2
        boundaries[-1] = 1
        self.weights = boundaries[1:] - boundaries[:-1]

    def solve(self, sample):
        # integrate on grid
        integral = np.sum(sample(self.grid)*self.weights)
        return integral

def sampler(finegrid, coarsegrid=None):
    tp = 1001
    epsilon = 0.2

    xp = np.linspace(0, 1, tp)
    noise = rng.normal(0, epsilon, tp)
    fp = 1 + np.sin(2*np.pi*xp) + noise
    g = lambda x: np.interp(x, xp, fp)

    if coarsegrid is None:
        return g, None
    else:
        return g, g

def level_maker(fine, coarse, comm=MPI.COMM_WORLD):
    def level(number):
        levelpoints = [10, 20, 40]
        x = np.linspace(0, 1, levelpoints[number] + 1)
        return (x[:-1]+x[1:])/2

    if coarse < 0:
        return level(fine), None
    else:
        return level(fine), level(coarse)

def plotting():
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 1, 1001)
    g, _ = sampler(None)
    plt.plot(xs, g(xs), linewidth=0.5)

    xf, xc = level_maker(0, 1)
    plt.plot(xc, g(xc), 'x-')
    plt.plot(xf, g(xf), '+-')
    plt.show()

# Levels and repetitions
levels = 3
repetitions = [100, 14, 2]
MLMCprob = MLMC_Problem(SimpleProblem, sampler, level_maker)
MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
estimate = MLMCsolv.solve()

print(estimate)

reps = levels*repetitions[-1]
xf, _ = level_maker(2, -1)
prob = SimpleProblem(xf)
total = 0
for _ in range(reps):
    total += prob.solve(sampler(None)[0])
print(total/reps)

