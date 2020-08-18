from firedrake import *
from randomgen import RandomGenerator, MT19937
import json
import matplotlib.pyplot as plt

from MLMCv5 import MLMC_Solver, MLMC_Problem

rg = RandomGenerator(MT19937(12345))

def samp(lvl_f, lvl_c):
    ans = 20*rg.random_sample()
    return ans, ans


def lvl_maker(level_f, level_c):
    coarse_mesh = UnitSquareMesh(10, 10)
    hierarchy = MeshHierarchy(coarse_mesh, level_f, 1)
    if level_c < 0:
        return hierarchy[level_f], None
    else:
        return hierarchy[level_f], hierarchy[level_c]


class problemClass:
    """
    Needs to take an integer initialisation argument to define the level (0 - L)
    Needs to have a .solve() method which takes a sample as an argument and returns
    a scalar solution
    """
    def __init__(self, level_obj):
        
        self._V = FunctionSpace(level_obj, "Lagrange", 4)
        self._sample = Constant(0)
        self._uh = Function(self._V)
        self._vs = self.initialise_problem()
    
    def solve(self, sample):
        self._sample.assign(Constant(sample))
        self._vs.solve()
        return assemble(Constant(0.5) * dot(self._uh, self._uh) * dx)
    
    # HELPER
    def initialise_problem(self):
        u = TrialFunction(self._V)
        v = TestFunction(self._V)

        x, y = SpatialCoordinate(self._V.mesh())
        base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
        a = (dot(grad(v), grad(u)) + v * u) * dx

        bcs = DirichletBC(self._V, 0, (1,2,3,4))
        f = self._sample*base_f
        L = f * v * dx
        vp = LinearVariationalProblem(a, L, self._uh, bcs=bcs)
        return LinearVariationalSolver(vp, solver_parameters={'ksp_type': 'cg'})
    
   

def general_test():
    # Levels and repetitions
    levels = 3
    repetitions = [100, 50, 10]
    MLMCprob = MLMC_Problem(problemClass, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate = MLMCsolv.solve()
    print(estimate)
    evaluate_result(estimate[0])


def evaluate_result(result):
    with open("10_int.json") as handle:
        e_10 = json.load(handle)
    
    with open("100_int.json") as handle:
        e_100 = json.load(handle)
    
    with open("1000_int.json") as handle:
        e_1000 = json.load(handle)
    
    with open("10000_int.json") as handle:
        e_10000 = json.load(handle)
    
    with open("20000_int.json") as handle:
        e_20000 = json.load(handle)

    d_10 = result - e_10
    d_100 = result - e_100
    d_1000 = result - e_1000
    d_10000 = result - e_10000
    d_20000 = result - e_20000

    print("% difference from 10 sample MC: ",(d_10*100)/result)
    print("% difference from 100 sample MC: ",(d_100*100)/result)
    print("% difference from 1000 sample MC: ",(d_1000*100)/result)
    print("% difference from 10000 sample MC: ",(d_10000*100)/result)
    print("% difference from 20000 sample MC: ",(d_20000*100)/result)

    convergence_tests(result)

def convergence_tests(param = None):
    """
    Function which compares result to 10,000 sample MC 
    """
    with open("20000_list.json") as handle:
            results = json.load(handle)
    
    res2 = [sum(results[:i+1])/(i+1) for i in range(len(results))]
    #print(res2[0], results[0])
    fig, axes = plt.subplots()
    axes.plot([i for i in range(20000)], res2, 'r')
    if param != None:
        plt.axhline(y=param, color='b')
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()


if __name__ == '__main__':
    general_test()