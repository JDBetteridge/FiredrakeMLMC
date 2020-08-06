from firedrake import *
from randomgen import RandomGenerator, MT19937
import json
import matplotlib.pyplot as plt
from matern import matern 

import time

from MLMCv5 import MLMC_Solver, MLMC_Problem, do_MC

#rg = RandomGenerator(MT19937(12345))

def samp(lvl_f, lvl_c):
    start = time.time()
    samp_c = None
    samp_f = matern(lvl_f, mean=1, variance=0.2, correlation_length=0.1)

    if lvl_c != None:
        samp_c = Function(lvl_c)
        inject(samp_f, samp_c)

    print("samp time: {}".format(time.time() - start))

    return samp_f, samp_c


def lvl_maker(level_f, level_c):
    coarse_mesh = UnitSquareMesh(20, 20)
    hierarchy = MeshHierarchy(coarse_mesh, level_f, 1)
    if level_c < 0:
        return FunctionSpace(hierarchy[level_f], "CG", 2), None
    else:
        return FunctionSpace(hierarchy[level_f], "CG", 2), \
        FunctionSpace(hierarchy[level_c], "CG", 2)


class problemClass:
    """
    Needs to take an integer initialisation argument to define the level (0 - L)
    Needs to have a .solve() method which takes a sample as an argument and returns
    a scalar solution
    """
    def __init__(self, level_obj):
        
        self._V = level_obj

        self._sample = Function(self._V)
        self._qh = Function(self._V)
        self._vs = self.initialise_problem()
    
    def solve(self, sample):
        #print(self._V.mesh())
        self._sample.assign(sample)
        self._vs.solve()
        print(self._V.mesh())
        return assemble(dot(self._qh, self._qh) * dx)
    
    # HELPER
    def initialise_problem(self):

        q = TrialFunction(self._V)
        p = TestFunction(self._V)
        u = self._sample

        a = inner(exp(u)*grad(q), grad(p))*dx
        L = inner(Constant(1.0), p)*dx
        bcs = DirichletBC(self._V, Constant(0.0), (1,2,3,4))

        vp = LinearVariationalProblem(a, L, self._qh, bcs=bcs)
        solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
        
        return LinearVariationalSolver(vp, solver_parameters=solver_param)

def general_test():
    # Levels and repetitions
    levels = 4
    repetitions = [100, 100, 100, 100]
    MLMCprob = MLMC_Problem(problemClass, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate = MLMCsolv.solve()
    print(estimate)
    evaluate_result(estimate)


def test_MC(reps, mesh_dim):
    mesh = UnitSquareMesh(mesh_dim, mesh_dim)
    V = FunctionSpace(mesh, "CG", 2)

    string = "randomfieldMC_{}r_{}dim".format(reps, mesh_dim)

    results = do_MC(problemClass, reps, V, samp)
    with open(string+'.json', 'w') as f:
        json.dump(results, f)

    res2 = [sum(results[:i+1])/(i+1) for i in range(len(results))]
    #print(res2[0], results[0])
    fig, axes = plt.subplots()
    axes.plot([i for i in range(reps)], res2, 'r')
    
    #plt.show()





def evaluate_result(result):

    with open("randomfieldMC_500r_160dim.json") as handle:
        e_500 = json.load(handle)
    d_500 = result - (sum(e_500)/len(e_500))

    print("% difference from 500 sample MC: ",(d_500*100)/result)


    convergence_tests(result)

def convergence_tests(param = None):

    # Function which compares result to 10,000 sample MC 
    
    with open("randomfieldMC_500r_160dim.json") as handle:
            results1 = json.load(handle)
    
    with open("randomfieldMC_500r_10dim.json") as handle:
            results2 = json.load(handle)
    
    with open("randomfieldMC_500r_20dim.json") as handle:
            results3 = json.load(handle)
    
    with open("randomfieldMC_500r_40dim.json") as handle:
            results4 = json.load(handle)

    with open("randomfieldMC_500r_80dim.json") as handle:
            results5 = json.load(handle)
    
    res1 = [sum(results1[:i+1])/(i+1) for i in range(len(results1))]
    res2 = [sum(results2[:i+1])/(i+1) for i in range(len(results2))]
    res3 = [sum(results3[:i+1])/(i+1) for i in range(len(results3))]
    res4 = [sum(results4[:i+1])/(i+1) for i in range(len(results4))]
    res5 = [sum(results5[:i+1])/(i+1) for i in range(len(results5))]

    #print(res2[0], results[0])
    fig, axes = plt.subplots()
    x = [i for i in range(500)]
    axes.plot(x, res2, 'y', label="10x10") 
    axes.plot(x, res3, 'orange', label="20x20") 
    axes.plot(x, res4, 'r', label="40x40")
    axes.plot(x, res5, 'brown', label="80x80")
    axes.plot(x, res1, 'k', label="160x160")
    if param != None:
        plt.axhline(y=param, color='b', label="MLMC Solution")
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    axes.legend()
    axes.set_ylabel('Solution')
    axes.set_xlabel('Repititions')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()



if __name__ == '__main__':
    general_test()
    #test_MC(500, 20)
    #test_MC(500, 80)
    #convergence_tests()