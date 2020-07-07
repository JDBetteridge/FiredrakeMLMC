import matplotlib.pyplot as plt
from firedrake import *
from randomgen import RandomGenerator, MT19937
import time
import json


def MC_scalar(mesh_size, iterations, mult):
    start = time.time()

    mesh = UnitSquareMesh(mesh_size, mesh_size)
    sample = Constant(0)
    V = FunctionSpace(mesh, "Lagrange", 4)
    uh = Function(V)
    vp = newProblem(mesh, sample, uh)

    rg = RandomGenerator(MT19937(12345))
    solutions = []

    for i in range(iterations):
        print("Sample {} of {}".format(i+1, iterations))
        sample.assign(Constant(mult*rg.random_sample()))

        vp.solve()
        energy = assemble(Constant(0.5) * dot(uh, uh) * dx)
        
        solutions.append(energy)
        estimate = sum(solutions)/(i+1)
   
    end = time.time()
    print("Runtime: ", end - start, "s") 

    fig, axes = plt.subplots()
    #axes.plot([i for i in range(iterations)], results, 'r')
    axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()
    
    FILE_NAME = "20000_int.json"
    with open(FILE_NAME, "w") as handle:
        json.dump(estimate, handle)
    
    FILE_NAME = "20000_list.json"
    with open(FILE_NAME, "w") as handle:
        json.dump(solutions, handle)
    

    return 0

# ============================================================================ #

def prob(mesh, alpha=1):
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))
    f = Constant(alpha)*base_f
    L = f * v * dx

    uh = Function(V)

    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})

    energy = assemble(Constant(0.5) * dot(uh, uh) * dx)
    return energy

def prob2(mesh, alpha=1, setup=True):
    output = [False for i in range(4)]
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    
    a = (dot(grad(v), grad(u)) + v * u) * dx
    output[0] = a
    output[2] = DirichletBC(V, 0, (1,2,3,4))
    output[3] = Function(V)

  
    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    f = Constant(alpha)*base_f
    L = f * v * dx
    output[1] = L

    

    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})


rg = RandomGenerator(MT19937(12345))
def samp():
    return 20*rg.random_sample()


def MLMC_general_scalar(pde_problem, scalar_problem, sampler, starting_mesh, levels, repititions, isEval=True):
    """
    arg: problem (func) - function of problem which takes 2 arguments: mesh and 
                          and a random sample, returns scalar solution
         sampler (func) - no argument function which returns a random sample
         levels (int) - number of levels in MLMC
         repititions (list) - repititions at each level starting at coarsest
         isEval (bool) - whether or not evaluation should be run on result
    output: Estimate of value
    """
    start = time.time()

    assert len(repititions) == levels, \
    ("The levels arguement is not equal to the number of entries in repititions")

    solver = MLMC_Solver(pde_problem, starting_mesh, sampler, levels, scalar_problem)

    # Iterate through each level in hierarchy
    for i in range(levels):
        print("LEVEL {} - {} Samples".format(i+1, repititions[i]))

        solver.newLevel(i) # Create P_level obj in soln list
        
        # Sampling now begins
        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))

            solver.addTerm(i) # Calculate result from sample

        # This corresponds to the inner sum in the MLMC eqn.
        solver.averageLevel(i)
    
    # Outer sum in MLMC eqn.
    estimate = solver.sumAllLevels()
    
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
    repititions = [1000, 200, 10]
    
    # Creating base coarse mesh and function spaceokay 
    coarse_mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(coarse_mesh, "Lagrange", 4)

    #estimate = MLMC_general(V, levels, repititions, samples, True)
    estimate = MLMC_general_scalar(newProblem, scalarProblem, samp, V, levels, repititions, True)




class MLMC_Solver:
    def __init__(self, pde_problem, coarse_mesh, sampler, levels, scalar_problem):
        self.pde_problem = pde_problem
        self.scalar_problem = scalar_problem
        self._coarse_V = coarse_mesh
        self.sampler = sampler

        # List with entry for each level
        # When entry == False level calculation has not begun
        # When type(entry) == P_level obj calculation in progress (summing terms)
        # When type(entry) == float obj calculation on that level completed
        self._level_list = [False for i in range(levels)]

        self._result = None

    def newLevel(self, level):

        self._level_list[level] = P_level(self._coarse_V, self.pde_problem, self.scalar_problem, self.sampler, level)

    def addTerm(self, level):
        self._level_list[level].calculate_term()
    
    def averageLevel(self, level):
        self._level_list[level] = self._level_list[level].get_average()

    def sumAllLevels(self):
        assert all(isinstance(x, float) for x in self._level_list)
        self._result = sum(self._level_list)
        return self._result
    
    def eval_result(self):
        
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

        d_10 = self._result - e_10
        d_100 = self._result - e_100
        d_1000 = self._result - e_1000
        d_10000 = self._result - e_10000
        d_20000 = self._result - e_20000

        print("% difference from 10 sample MC: ",(d_10*100)/self._result)
        print("% difference from 100 sample MC: ",(d_100*100)/self._result)
        print("% difference from 1000 sample MC: ",(d_1000*100)/self._result)
        print("% difference from 10000 sample MC: ",(d_10000*100)/self._result)
        print("% difference from 20000 sample MC: ",(d_20000*100)/self._result)

        convergence_tests(self._result)



class P_level:
    
    def __init__(self, coarse_V, pde_problem, scalar_problem, sampler, level):
        # Assignments
        self.pde_problem = pde_problem
        self.scalar_problem = scalar_problem
        self._level = level
        self.sampler = sampler

        # Create Hierarchy
        self._hierarchy = MeshHierarchy(coarse_V.mesh(), level, 1)

        self._sample_counter = 0 # Needed for divison in get_average()
        self._value = None # Stores result
        self._sample = Constant(0) # Stores current sample

        family = coarse_V.ufl_element().family()
        degree = coarse_V.ufl_element().degree()
        # Create variational problem(s)
        self.initialise_problem(family, degree)
        
    
    def initialise_problem(self, family, degree):
        # Set up fine variational problem with sample which varies
        V_f = FunctionSpace(self._hierarchy[self._level], family, degree)
        self._uh_f = Function(V_f)
        self._vs_f = self.pde_problem(self._hierarchy[self._level], self._sample, self._uh_f)
        
        # Set up coarse variational problem with sample which varies
        if self._level-1 >= 0:
            V_c = FunctionSpace(self._hierarchy[self._level-1], family, degree)
            self._uh_c = Function(V_c)
            self._vs_c = self.pde_problem(self._hierarchy[self._level-1], self._sample, self._uh_c) 


    def get_average(self):
        self._hierarchy = None # For memory conservation clear hierarchy
        return self._value/self._sample_counter
        

    def calculate_term(self):
        """
        Calculates result from new sample and adds it to _value. This is 
        equivalent to inner sum in MLMC equation.
        """
        if self._value == None:
            self._value = 0

        # Generate sample and call problem fuction on sample
        self._sample.assign(Constant(self.sampler()))
        self._vs_f.solve()
        
        e_f = self.scalar_problem(self._uh_f)

        if self._level-1 >= 0:  
            self._vs_c.solve()
            e_c = self.scalar_problem(self._uh_c)
            self._value +=  e_f - e_c
        else:
            self._value += e_f
        self._sample_counter += 1
        return 0

# ============================================================================ #

def newProblem(mesh, alpha, uh):
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))
    f = alpha*base_f
    L = f * v * dx
    vp = LinearVariationalProblem(a, L, uh, bcs=bcs)

    return LinearVariationalSolver(vp, solver_parameters={'ksp_type': 'cg'})

def scalarProblem(uh):
    return assemble(Constant(0.5) * dot(uh, uh) * dx)

def convergence_tests(param = None):
    """
    Function which compares result to 10,000 sample MC 
    """
    with open("20000_list.json") as handle:
            results = json.load(handle)
    
    res2 = [sum(results[:i])/(i+1) for i in range(len(results))]
    fig, axes = plt.subplots()
    axes.plot([i for i in range(20000)], res2, 'r')
    if param != None:
        plt.axhline(y=param, color='b')
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()

if __name__ == '__main__':
    #print(MC_scalar(40, 20000, 20))
    general_test()
    #convergence_tests()

    