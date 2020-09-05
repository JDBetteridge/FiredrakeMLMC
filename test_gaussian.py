from firedrake import *
from randomgen import RandomGenerator, MT19937
import json
#import matplotlib.pyplot as plt

import matplotlib.pylab as plt
from matplotlib import rc
from mpi4py import MPI
import time

from MLMCv6 import MLMC_Solver, MLMC_Problem, do_MC

rg = RandomGenerator(MT19937(12345))

def samp(lvl_f, lvl_c):
    ans = 20*rg.random_sample()
    #print(ans)
    return ans, ans


def lvl_maker(level_f, level_c, comm=MPI.COMM_WORLD):
    coarse_mesh = UnitSquareMesh(20, 20, comm=comm)
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
        #print(self._V.mesh().num_faces())
        self._sample = Constant(0)
        self._uh = Function(self._V)
        self._vs = self.initialise_problem()
    
    def solve(self, sample):
        sample = 5
        print(sample)
        self._uh.assign(0)
        self._sample.assign(Constant(sample))
        self._vs.solve()
        print(assemble(dot(self._uh, self._uh) * dx))
        return assemble(dot(self._uh, self._uh) * dx)
    
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
    levels = 4
    repetitions = [1000, 100, 50, 10]
    MLMCprob = MLMC_Problem(problemClass, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    s = time.time()
    estimate = MLMCsolv.solve()
    print(estimate)
    print("Total time: {}".format(time.time()-s))
    evaluate_result(estimate)


def evaluate_result(result):
    lvl_res = result[1]
    result = result[0]
    with open("Gaussian_1000r_20dim.json") as handle:
        e_20 = json.load(handle)
    
    with open("Gaussian_1000r_40dim.json") as handle:
        e_40 = json.load(handle)

    with open("Gaussian_1000r_80dim.json") as handle:
        e_80 = json.load(handle) 
    
    with open("Gaussian_1000r_160dim.json") as handle:
        e_160 = json.load(handle)


    d_20 = result - sum(e_20)/len(e_20)
    d_40 = result - sum(e_40)/len(e_40)
    d_80 = result - sum(e_80)/len(e_80)
    d_160 = result - sum(e_160)/len(e_160)

    print("% difference from 1000 sample 20x20 MC: ",(d_20*100)/result)
    print("% difference from 1000 sample 40x40 MC: ",(d_40*100)/result)
    print("% difference from 1000 sample 80x80 MC: ",(d_80*100)/result)
    print("% difference from 1000 sample 160x160 MC: ",(d_160*100)/result)

    #croci_convergence(lvl_res)
    convergence_tests(lvl_res)

def croci_convergence(level_res):
    levels = [1/20**2, 1/40**2, 1/80**2, 1/160**2, 1/320**2]
    fig, axes = plt.subplots()

    a = axes.plot(levels[1:], level_res[1:], '+', markersize=10, color='k', label=r'Results ($\nu = 1$)') 
    h2 = [0.03*i**2 for i in levels]

    axes.plot(levels[1:], h2[1:], '--', color='k', label=r'Theory $O(1/n^2)$') 
    
    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_ylabel(r'$\mathrm{\mathbb{E}} \left[\left\Vert q_\ell\right\Vert^2_{L^2} - \left\Vert q_{\ell-1}\right\Vert^2_{L^2}\right]$', fontsize=16)
    axes.set_xlabel(r'$1/n_\ell$', fontsize=16)
    plt.tight_layout()
    plt.style.use('classic')
    plt.legend(loc='best', prop={'size': 13}, numpoints=1)

    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')
    plt.show()

def convergence_tests(param = None):
    """
    Function which compares result to 10,000 sample MC 
    """
    with open("Gaussian_1000r_20dim.json") as handle:
        e_20 = json.load(handle)
    
    with open("Gaussian_1000r_40dim.json") as handle:
        e_40 = json.load(handle)

    with open("Gaussian_1000r_80dim.json") as handle:
        e_80 = json.load(handle) 
    
    with open("Gaussian_1000r_160dim.json") as handle:
        e_160 = json.load(handle)
    

    res20 = [sum(e_20[:i+1])/(i+1) for i in range(len(e_20))]
    res40 = [sum(e_40[:i+1])/(i+1) for i in range(len(e_40))]
    res80 = [sum(e_80[:i+1])/(i+1) for i in range(len(e_80))]
    res160 = [sum(e_160[:i+1])/(i+1) for i in range(len(e_160))]
    show_results(res20, res40, res80, res160, param)
    #limit = res160[-1]
    #convergence(res20, res40, res80, res160, limit)
    """
    #print(res2[0], results[0])
    fig, axes = plt.subplots()
    axes.plot([i for i in range(1000)], res2, 'r')
    if param != None:
        sums = [sum(param[:(i+1)]) for i in range(len(param))]
        plt.axhline(y=sums[0], color='b')
        plt.axhline(y=sums[1], color='g')
        plt.axhline(y=sums[2], color='gold')
        plt.axhline(y=sums[3], color='k')
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()
    """

def convergence(res, res2, res3, res4, limit):

    logN, error = convergence_check(res, limit)
    logN2, error2 = convergence_check(res2, limit)
    logN3, error3 = convergence_check(res3, limit)
    logN4, error4 = convergence_check(res4, limit)

    halfx = [(error[0])*i**(-0.5) for i in logN]
    halfx2 = [(error4[0]*0.1)*i**(-0.5) for i in logN]

    fig, axes = plt.subplots()

    axes.plot(logN, error, 'gold', label=r'20x20') 
    #axes.plot(logN3, error3, 'orange', label=r'40x40')
    #axes.plot(logN2, error2, 'r', label=r'80x80') 
    #axes.plot(logN, error, 'brown', label=r'160x160') 
    axes.plot(logN4, error4, 'k', label=r'160x160') 
    

    axes.plot(logN, halfx, '--', color='k', label=r'$O(N^{-1/2})$') 
    axes.plot(logN, halfx2, '--', color='k') 
    
    axes.set_ylabel(r'$\mathrm{\mathbb{E}} \left[\left\Vert q_L \right\Vert^2_{L^2} \right] - \mathrm{\mathbb{E}} \left[\left\Vert q_\ell \right\Vert^2_{L^2} \right]$', fontsize=14)
    axes.set_xlabel(r'Repetitions, $N$', fontsize=13)
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt.style.use('classic')
    plt.legend(loc="best", prop={'size': 10})
    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')

    plt.tight_layout()
    plt.show()

def convergence_check(res, limit):
    error = [abs(limit-element) for element in res]
    logN = [i+1 for i in range(len(res))]
    return logN, error

def show_results(res1, res2, res3, res4, param):
    
    fig, axes = plt.subplots()
    x2 = [i for i in range(1000)]
    #axes.plot(x2, res1, 'gold', label="20x20") 
    #axes.plot(x2, res2, 'orange', label="40x40") 
    #axes.plot(x2, res3, 'r', label="80x80")
    axes.plot(x2, res4, 'k', label=r"$\ell = 3 \; (160\times160)$")
    if param != None:
        plt.axhline(y=param[0], linestyle='--' ,color='k', label=r"$\mathrm{\mathbb{E}} \left[\Vert u\Vert^2_{L^2} \right]_{MLMC}$")
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.style.use('classic')
    axes.legend(loc="best", prop={'size': 13})
    
    axes.set_ylabel(r'$\mathrm{\mathbb{E}} \left[\Vert u \Vert^2_{L^2} \right]$', fontsize=14)
    axes.set_xlabel(r'Repetitions, $N$', fontsize=13)
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')
    
    plt.tight_layout()
    plt.show()

def test_MC(reps, mesh_dim):
    mesh = UnitSquareMesh(mesh_dim, mesh_dim)
    V = FunctionSpace(mesh, "CG", 2)

    string = "Gaussian_{}r_{}dim".format(reps, mesh_dim)

    results = do_MC(problemClass, reps, V, samp)
    with open(string+'.json', 'w') as f:
        json.dump(results, f)

    res2 = [sum(results[:i+1])/(i+1) for i in range(len(results))]
    fig, axes = plt.subplots()
    axes.plot([i for i in range(reps)], res2, 'r')
    plt.show()

def manual_test(samples):
    # made for 17 samples
    coarse = UnitSquareMesh(20,20)
    hierarchy = MeshHierarchy(coarse, 2, 1)
    level0 = problemClass(FunctionSpace(hierarchy[0], "CG", 2))
    level1 = problemClass(FunctionSpace(hierarchy[1], "CG", 2))
    level2 = problemClass(FunctionSpace(hierarchy[2], "CG", 2))

    level0_results = [level0.solve(samples[i]) for i in range(1)]
    level1_results = [[level1.solve(samples[i]), level0.solve(samples[i])] for i in range(1, 2)] 
    level2_results = [[level2.solve(samples[i]), level1.solve(samples[i])] for i in range (2, 3)]

    L0 = sum(level0_results)/len(level0_results)
    L1_sub = [i[0]-i[1] for i in level1_results]
    L1 = sum(L1_sub)/len(L1_sub)
    L2_sub = [i[0]-i[1] for i in level2_results]
    L2 = sum(L2_sub)/len(L2_sub)

    print((L0+L1+L2,[L0,L1,L2]))

if __name__ == '__main__':
    #general_test()
    #test_MC(1000,80)
    #rg = RandomGenerator(MT19937(12345))
    #ans = [20*rg.random_sample() for i2 in range(3)]
    #manual_test(ans)
    #convergence_tests()
    test_MC(100, 20)