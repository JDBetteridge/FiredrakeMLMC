import matplotlib.pyplot as plt
from firedrake import *
from randomgen import RandomGenerator, MT19937
import time
import json


def MC_scalar(mesh_size, iterations, mult):
    start = time.time()

    mesh = UnitSquareMesh(mesh_size, mesh_size)

    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    rg = RandomGenerator(MT19937(12345))
    solutions = []

    f = Constant(mult/2)*base_f
    L = f*v*dx
    u_real = Function(V)
    solve(a == L, u_real, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
    e_real = assemble(Constant(0.5) * dot(u_real, u_real) * dx)

    results = []
    for i in range(iterations):
        print("Sample {} of {}".format(i+1, iterations))
        rand_multiplier = Constant(mult*rg.random_sample())
        f = rand_multiplier*base_f

        L = f * v * dx 

        uh = Function(V)

        solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})

        energy = assemble(Constant(0.5) * dot(uh, uh) * dx)
        
        solutions.append(energy)
        estimate = sum(solutions)/(i+1)
        error = ((estimate - e_real)*100)/estimate
        results.append(error)

    

    
    end = time.time()
    print("Runtime: ", end - start, "s") 



    fig, axes = plt.subplots()
    #axes.plot([i for i in range(iterations)], results, 'r')
    axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()

    FILE_NAME = "1000_int.json"
    with open(FILE_NAME, "w") as handle:
        json.dump(estimate, handle)
    
    FILE_NAME = "1000_list.json"
    with open(FILE_NAME, "w") as handle:
        json.dump(solutions, handle)

    return results

# ============================================================================ #

def prob(mesh, alpha=1):
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    f = alpha*base_f
    L = f * v * dx

    uh = Function(V)

    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
    energy = assemble(Constant(0.5) * dot(uh, uh) * dx)
    return energy

rg = RandomGenerator(MT19937(12345))
def samp():
    return 20*rg.random_sample()


def MLMC_general_scalar(problem, sampler, starting_mesh, levels, repititions, isEval=True):
    """
    arg: levels - number of levels in MLMC
         repititions (list) - repititions at each level starting at coarsest
         samples (list) - list of all samples
         isEval (bool) - whether or not evaluation should be run on result
    output: Estimate of value
    """
    start = time.time()

    assert len(repititions) == levels, \
    ("The levels arguement is not equal to the number of entries in the iterable")

    solver = MLMC_Solver(problem, starting_mesh, sampler, levels)

    # Iterate through each level in hierarchy
    for i in range(levels):
        print("LEVEL {} - {} Samples".format(i+1, repititions[i]))

        # By this point function/ function spaces have been set up
        # Sampling now begins
        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))
            
            solver.addTerm(i)

        
        # This sum corresponds to the inner sum in the MLMC eqn.
        solver.calculateInnerSum()
    
    # Outer sum in MLMC eqn.
    estimate = solver.calculateOuterSum()
    
    end = time.time()
    print("Runtime: ", end - start, "s")

    if isEval:
        solver.eval_result()

    return estimate


def eval_soln(estimate, mult, mesh_f):

    uh_true = prob(mesh_f, mult)

    difference = assemble(estimate - uh_true)

    fig, axes = plt.subplots()
    collection = tripcolor(difference, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)
    plt.show()

    print(errornorm(estimate, uh_true))


def general_test():
    # Levels and repititions
    levels = 3
    repititions = [50, 10, 5]
    
    # Creating base coarse mesh and function space
    coarse_mesh = UnitSquareMesh(10, 10)

    #estimate = MLMC_general(V, levels, repititions, samples, True)
    estimate = MLMC_general_scalar(prob, samp, coarse_mesh, levels, repititions, True)




class MLMC_Solver:
    def __init__(self, problem, starting_mesh, sampler, levels):
        self.problem = problem
        self._hierarchy = MeshHierarchy(starting_mesh, levels-1, 1)
        self.sampler = sampler

        self._solutions = []
        self._sub_solutions = []

        self._result = None

    def addTerm(self, level):
        term = P_term(self._hierarchy, self.problem, self.sampler, level)
        term.calculate()
        self._sub_solutions.append(term)
    
    def calculateInnerSum(self):
        level_result = sum(self._sub_solutions)/len(self._sub_solutions)
        self._solutions.append(level_result)
        self._sub_solutions = []

    def calculateOuterSum(self):
        self._result = sum(self._solutions).get_value()
        return self._result
    
    def eval_result(self):
        
        with open("10_int.json") as handle:
            e_10 = json.load(handle)
        
        with open("100_int.json") as handle:
            e_100 = json.load(handle)
        
        with open("1000_int.json") as handle:
            e_1000 = json.load(handle)

        d_10 = self._result - e_10
        d_100 = self._result - e_100
        d_1000 = self._result - e_1000

        print("% difference from 10 sample MC: ",(d_10*100)/self._result)
        print("% difference from 100 sample MC: ",(d_100*100)/self._result)
        print("% difference from 1000 sample MC: ",(d_1000*100)/self._result)


class P_term:
    
    def __init__(self, hierarchy, problem, sampler, level):
        self.problem = problem
        self._level = level
        self._hierarchy = hierarchy
        self.sampler = sampler

        self._value = None

    
    def __add__(self, other):
        assert self._value != None and other._value != None, \
        ("Both terms in sum need to have been calculated before they can be summed")

        self._value += other._value
        return self
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __truediv__(self, other):
        assert self._value != None, \
        ("P_term object needs to be calculated before division can be carried out")

        self._value = self._value / other
        return self
    
    def get_value(self):
        return self._value

    def calculate(self):
        assert self._value == None, ("Error: Already calculated")

        mesh_f = self._hierarchy[self._level]
        # Call problem fuction
        sample = self.sampler()
        e_f = self.problem(mesh_f, sample)
    
        if self._level-1 >= 0:  
            mesh_c = self._hierarchy[self._level-1]
            e_c = self.problem(mesh_c, sample)
                
            self._value =  e_f - e_c
        else:
            self._value = e_f
        return 0

if __name__ == '__main__':
    #print(MC_scalar(40, 1000, 20))
    general_test()
