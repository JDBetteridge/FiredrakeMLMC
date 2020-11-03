import json
import math
import time

import matplotlib.pyplot as plt

from firedrake import *
from mpi4py import MPI
from randomgen import RandomGenerator, MT19937

from mlmcparagen import MLMC_Solver, MLMC_Problem, do_MC
from mlmcparagen.randomfield.matern import matern

plt.rcParams['mathtext.fontset'] = 'stix'

def samp(lvl_f, lvl_c):
    start = time.time()
    samp_c = None
    samp_f = matern(lvl_f, mean=1, variance=0.2, correlation_length=0.2, smoothness=1)

    if lvl_c != None:
        samp_c = Function(lvl_c)
        inject(samp_f, samp_c)

    #print("samp time: {}".format(time.time() - start))

    return samp_f, samp_c


def lvl_maker(level_f, level_c, comm=MPI.COMM_WORLD):
    coarse_mesh = UnitSquareMesh(320, 320, comm=comm)
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
        self._qh.assign(0)
        self._sample.assign(sample)
        s = time.time()
        self._vs.solve()
        firedrake.logging.warning("sample time: {}".format(time.time()-s))
        return assemble(dot(self._qh, self._qh) * dx)

    # HELPER
    def initialise_problem(self):

        q = TrialFunction(self._V)
        p = TestFunction(self._V)

        a = inner(exp(self._sample)*grad(q), grad(p))*dx
        L = inner(Constant(1.0), p)*dx
        bcs = DirichletBC(self._V, Constant(0.0), (1,2,3,4))

        vp = LinearVariationalProblem(a, L, self._qh, bcs=bcs,
        constant_jacobian=False)
        solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}

        return LinearVariationalSolver(vp, solver_parameters=solver_param)

def general_test_para():
    # Levels and repetitions
    s = time.time()
    levels = 2
    repetitions = [64, 64]
    limits = [1, 1]
    MLMCprob = MLMC_Problem(problemClass, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions, comm=MPI.COMM_WORLD, comm_limits=limits)
    estimate, lvls = MLMCsolv.solve()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Total Time Taken: ", time.time() - s)
        print(estimate)
        print(list(lvls))
        #with open('MLMC_100r_5lvl_20dim_3nu.json', 'w') as f:
        #    json.dump(list(lvls), f)

def general_test_serial():
    # Levels and repetitions
    s = time.time()
    levels = 2
    repetitions = [100, 100]
    MLMCprob = MLMC_Problem(problemClass, samp, lvl_maker)
    MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
    estimate, lvls = MLMCsolv.solve()
    print("Total Time Taken: ", time.time() - s)
    print(estimate)
    print(lvls)

    #evaluate_result(estimate)


def t_MC(reps, mesh_dim):
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

    plt.show()



def croci_convergence():
    with open("MLMC_100r_5lvl_20dim_1nu.json") as handle:
        level_res1 = json.load(handle)
        level_res1 = list(map(abs, level_res1))
    #print(level_res1)
    with open("MLMC_100r_5lvl_20dim_3nu.json") as handle:
        level_res3 = json.load(handle)
        level_res3 = list(map(abs, level_res3))
    #print(level_res3)

    levels = [math.sqrt(2)/20, math.sqrt(2)/40, math.sqrt(2)/80, math.sqrt(2)/160, math.sqrt(2)/320]
    print(levels)
    fig, axes = plt.subplots()


    a = axes.plot(levels[1:], level_res1[1:], '+', markersize=10, color='k', label=r'Results ($\nu = 1$)')
    b = axes.plot(levels[1:], level_res3[1:], 'x', markersize=8, color='k', label=r'Results ($\nu = 3$)')
    h2 = [0.013*i**2 for i in levels]
    h3 = [0.0075*i**4 for i in levels]

    axes.plot(levels[1:], h2[1:], '--', color='k', label=r'Theory $O(h^2)$')
    axes.plot(levels[1:], h3[1:], ':', color='k', label=r'Theory $O(h^4)$')
    #print(level_res3[1:])
    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_ylabel(r'$|\mathrm{\mathbb{E}} \left[\Vert q_\ell\Vert^2_{L^2} - \Vert q_{\ell-1}\Vert^2_{L^2}\right]|$', fontsize=16)
    axes.set_xlabel(r'$h_\ell$', fontsize=16)
    axes.set_ylim((10**-13, 10**-4))
    #axes.set_ylim((10**-13, 10**-4))
    plt.tight_layout()
    plt.style.use('classic')
    plt.legend(loc='best', prop={'size': 13}, numpoints=1)

    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')
    plt.show()



def evaluate_result(result):

    with open("randomfieldMC_500r_160dim.json") as handle:
        e_500 = json.load(handle)
    d_500 = result - (sum(e_500)/len(e_500))

    print("% difference from 500 sample MC: ",(d_500*100)/result)


    convergence_tests(result)

def convergence_tests(param = None):

    # Function which compares result to 10,000 sample MC
    with open("MLMCtest_5lvl_1nu.json") as handle:
            param = json.load(handle)
            param = sum(param)

    with open("randomfieldMC_1000r_160dim.json") as handle:
            results1 = json.load(handle)

    with open("randomfieldMC_1000r_320dim.json") as handle:
            results2 = json.load(handle)

    with open("randomfieldMC_1000r_20dim.json") as handle:
            results3 = json.load(handle)
    #print(results3)

    with open("randomfieldMC_1000r_40dim.json") as handle:
            results4 = json.load(handle)

    with open("randomfieldMC_1000r_80dim.json") as handle:
            results5 = json.load(handle)

    result = param
    d_20 = result - sum(results3)/len(results3)
    d_40 = result - sum(results4)/len(results4)
    d_80 = result - sum(results5)/len(results5)
    d_160 = result - sum(results1)/len(results1)
    d_320 = result - sum(results2)/len(results2)

    print("% difference from 1000 sample 20x20 MC: ",(d_20*100)/result)
    print("% difference from 1000 sample 40x40 MC: ",(d_40*100)/result)
    print("% difference from 1000 sample 80x80 MC: ",(d_80*100)/result)
    print("% difference from 1000 sample 160x160 MC: ",(d_160*100)/result)
    print("% difference from 1000 sample 320x320 MC: ",(d_320*100)/result)

    res160 = [sum(results1[:i+1])/(i+1) for i in range(len(results1))]
    res320 = [sum(results2[:i+1])/(i+1) for i in range(len(results2))]
    res20 = [sum(results3[:i+1])/(i+1) for i in range(len(results3))]
    res40 = [sum(results4[:i+1])/(i+1) for i in range(len(results4))]
    res80 = [sum(results5[:i+1])/(i+1) for i in range(len(results5))]


    #print(res2[0], results[0])
    limit = res320[-1]
    show_results(res20, res40, res80, res160, res320, param)
    #convergence(res160, res80, res40, res20, res320, limit)

def show_results(res1, res2, res3, res4, res5, param):
    fig, axes = plt.subplots()
    x2 = [i for i in range(1000)]
    #axes.plot(x2, res1, 'gold', label="20x20")
    #axes.plot(x2, res2, 'orange', label="40x40")
    axes.plot(x2, res3, color='#009c00', label=r"$\ell = 2 \; (80\times80)$")
    axes.plot(x2, res4, color='#007a00', label=r"$\ell = 3 \; (160\times160)$")
    axes.plot(x2, res5, 'k', label=r"$\ell = 4 \; (320\times320)$")
    if param != None:
        plt.axhline(y=param, linestyle='--' ,color='k', label=r"$\mathrm{\mathbb{E}} \left[\Vert u\Vert^2_{L^2} \right]_{MLMC}$")
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.style.use('classic')
    axes.legend(loc="best", prop={'size': 13})

    axes.set_ylabel(r'$\mathrm{\mathbb{E}} \left[\left\Vert q\right\Vert^2_{L^2} \right]$', fontsize=14)
    axes.set_xlabel(r'Repetitions, $N$', fontsize=13)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')

    plt.tight_layout()
    plt.show()

def convergence(res, res2, res3, res4, res5, limit):

    logN, error = convergence_check(res, limit)
    logN2, error2 = convergence_check(res2, limit)
    logN3, error3 = convergence_check(res3, limit)
    logN4, error4 = convergence_check(res4, limit)
    logN5, error5 = convergence_check(res5, limit)


    halfx = [(error4[0])*i**(-0.5) for i in logN]
    halfx2 = [(error5[0]*0.1)*i**(-0.5) for i in logN]

    fig, axes = plt.subplots()

    axes.plot(logN4, error4, color='#00ea00', label=r'$\ell = 0 \; (20\times20)$')
    #axes.plot(logN3, error3, 'orange', label=r'40x40')
    #axes.plot(logN2, error2, 'r', label=r'80x80')
    #axes.plot(logN, error, 'brown', label=r'160x160')
    axes.plot(logN5, error5, 'k', label=r'$\ell = 4 \; (320\times320)$')


    axes.plot(logN, halfx, '--', color='k', label=r'$O(N^{-1/2})$')
    axes.plot(logN, halfx2, '--', color='k')

    axes.set_ylabel(r'$\mathrm{\mathbb{E}} \left[\Vert q_L \Vert^2_{L^2} \right] - \mathrm{\mathbb{E}} \left[\Vert q_\ell \Vert^2_{L^2} \right]$', fontsize=14)
    axes.set_xlabel(r'Repetitions, $N$', fontsize=13)
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt.style.use('classic')
    plt.legend(loc="best", prop={'size': 13})
    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')

    plt.tight_layout()
    plt.show()

def convergence_check(res, limit):
    error = [abs(limit-element) for element in res]
    logN = [i+1 for i in range(len(res))]
    return logN, error


def matern_tests():
    mesh = UnitSquareMesh(160, 160, comm=MPI.COMM_WORLD)
    hier = MeshHierarchy(mesh, 1, 1)
    Vs = [FunctionSpace(i, "CG", 2) for i in hier]
    for i in Vs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(i.dim(), i.comm.Get_size())

    """
    fig, axes = plt.subplots()
    collection = tripcolor(samp_f, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)

    fig, axes = plt.subplots()
    collection = tripcolor(samp_c, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)
    print(res_f - res_c)
    #plt.show()
    """





def test_MC():
    general_test_para()
    #general_test_serial()
    #test_MC(1000, 20)
    #test_MC(1000, 40)
    #test_MC(1000, 80)
    #test_MC(1000, 160)
    #test_MC(1000, 320)
    #convergence_tests()
    #croci_convergence()
    #matern_tests()

    #with open("randomfieldMC_1000r_20dim.json") as handle:
    #    results3 = json.load(handle)

    #res20 = [sum(results3[:i+1])/(i+1) for i in range(len(results3))]
    #print(res20[-1])

