import matplotlib.pyplot as plt
from firedrake import *
from randomgen import RandomGenerator, MT19937
import time


"""
rg = RandomGenerator(MT19937(12345))
for i in range(10):
    print(rg.random_sample())

"""
def MC(mesh_size, iterations, mult):
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

    for i in range(iterations):
        print("Sample {} of {}".format(i+1, iterations))
        rand_multiplier = Constant(mult*rg.random_sample())
        f = rand_multiplier*base_f

        L = f * v * dx 

        uh = Function(V)

        solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
        solutions.append(uh)
    
    estimate = sum(solutions)/Constant(iterations)
    
    end = time.time()
    print("Runtime: ", end - start, "s") 

    f = Constant(mult/2)*base_f
    L = f*v*dx
    u_real = Function(V)
    solve(a == L, u_real, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
    
    e_real = assemble(Constant(0.5) * dot(u_real, u_real) * dx)

    est_f = interpolate(estimate, V)
    e_est = assemble(Constant(0.5) * dot(est_f, est_f) * dx)
    print(((e_est - e_real)*100)/e_est)

    difference = assemble(estimate-u_real)
    fig, axes = plt.subplots()
    collection = tripcolor(difference, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)

    plt.show()
    return errornorm(u_real, interpolate(estimate, V))

def MLMC_failed(mesh_sizes, repititions, mult):

    solutions = []
    rg = RandomGenerator(MT19937(12345))

    mesh_high = UnitSquareMesh(mesh_sizes[-1], mesh_sizes[-1])
    V_high = FunctionSpace(mesh_high, "Lagrange", 4)


    for i, size in enumerate(mesh_sizes):
        mesh_f = UnitSquareMesh(size, size)
        
        fig, axes = plt.subplots()
        triplot(mesh_f, axes=axes)
        axes.legend()


        V_f = FunctionSpace(mesh_f, "Lagrange", 4)
        u_f = TrialFunction(V_f)
        v_f = TestFunction(V_f)

        x_f, y_f = SpatialCoordinate(mesh_f)
        f_base_f = exp(-(((x_f-0.5)**2)/2) - (((y_f-0.5)**2)/2))
        a_f = (dot(grad(v_f), grad(u_f)) + v_f * u_f) * dx
        bcs_f = DirichletBC(V_f, 0, (1,2,3,4))

        if i-1 >= 0:  
            mesh_c = UnitSquareMesh(mesh_sizes[i-1], mesh_sizes[i-1]) 

            V_c = FunctionSpace(mesh_c, "Lagrange", 4)
            u_c = TrialFunction(V_c)
            v_c = TestFunction(V_c) 
            
            x_c, y_c = SpatialCoordinate(mesh_c)
            f_base_c = exp(-(((x_c-0.5)**2)/2) - (((y_c-0.5)**2)/2))
            a_c = (dot(grad(v_c), grad(u_c)) + v_c * u_c) * dx
            bcs_c = DirichletBC(V_c, 0, (1,2,3,4))


        
        sub_solutions = []

        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))
            
            rand_multiplier = Constant(mult*rg.random_sample())
            
            f_f = rand_multiplier*f_base_f
            L_f = f_f * v_f * dx 
            uh_f = Function(V_f)
            solve(a_f == L_f, uh_f, bcs=bcs_f, solver_parameters={'ksp_type': 'cg'})
        
            if i-1 >= 0:  
                f_c = rand_multiplier*f_base_c
                L_c = f_c * v_c * dx 
                uh_c = Function(V_c)
                solve(a_c == L_c, uh_c, bcs=bcs_c, solver_parameters={'ksp_type': 'cg'})
                print(type(Projector(uh_f - Projector(uh_c, V_f), V_high)))
                sub_solutions.append(Projector(uh_f - Projector(uh_c, V_f), V_high))
            else:
                u_new = Function(V_high)
                Projector(uh_f, u_new)
                sub_solutions.append(u_new)
                print(u_new.function_space())
        
        solutions.append(sum(sub_solutions)/Constant(repititions[i]))
    
    f = Constant(mult/2)*f_base_f
    L = f*v_f*dx
    u_real = Function(V_high)
    solve(a_f == L, u_real, bcs=bcs_f, solver_parameters={'ksp_type': 'cg'})

    estimate = sum(solutions)
    print(type(estimate))
    
    difference = assemble(estimate - u_real)
    fig, axes = plt.subplots()
    collection = tripcolor(difference, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)

    plt.show()
    
    return errornorm(u_real, interpolate(estimate, V_high))

def test1():
    
    coarse_mesh = UnitSquareMesh(10, 10)
    hierarchy = MeshHierarchy(coarse_mesh, 1, 1)

    mesh = hierarchy[0]

    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()


    V = FunctionSpace(mesh, "Lagrange", 4)
    print(V.mesh())
    print(V.ufl_element().family())
    print(V.ufl_element().degree())
    
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    L = f * v * dx 

    uh1 = Function(V)

    solve(a == L, uh1, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
    
    mesh = hierarchy[1]

    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()


    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    L = f * v * dx 

    uh2 = Function(V)

    solve(a == L, uh2, bcs=bcs, solver_parameters={'ksp_type': 'cg'})


    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()

    uh3 = Function(V)

    #solve(a == L, uh3, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
    
    print("1")
    prolong(uh1, uh3)
    print("2")
    
    difference = assemble(uh2 - uh3)
    fig, axes = plt.subplots()
    collection = tripcolor(difference, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)

    plt.show()
    return errornorm(uh2, uh3)


def MLMC_hier(coarse_size, levels, repititions, mult):
    """
    arg: coarse_size - dimension of face of Unit Square Mesh on coarsest level
         levels - number of levels in MLMC
         repititions (list) - repititions at each level starting at coarsest
         mult - multiplier on equartion to generate probability distribution
    output: errornorm between ground truth and result
    """
    start = time.time()

    solutions = []
    # Random Seed
    rg = RandomGenerator(MT19937(12345))

    # Initialise hierarchy
    coarse_mesh = UnitSquareMesh(coarse_size, coarse_size)
    hierarchy = MeshHierarchy(coarse_mesh, levels-1, 1)
    # Initialise function space at finest level
    V_high = FunctionSpace(hierarchy[-1], "Lagrange", 4)

    # Iterate through each level in hierarchy
    for i in range(len(hierarchy)):
        mesh_f = hierarchy[i]
        
        fig, axes = plt.subplots()
        triplot(mesh_f, axes=axes)
        axes.legend()

        # each iteration considers level l (_f) and l-1 (_c)
        # _f is the finer of the two levels
        V_f = FunctionSpace(mesh_f, "Lagrange", 4)
        u_f = TrialFunction(V_f)
        v_f = TestFunction(V_f)

        x_f, y_f = SpatialCoordinate(mesh_f)
        f_base_f = exp(-(((x_f-0.5)**2)/2) - (((y_f-0.5)**2)/2))
        a_f = (dot(grad(v_f), grad(u_f)) + v_f * u_f) * dx
        bcs_f = DirichletBC(V_f, 0, (1,2,3,4))

        if i-1 >= 0:  
            # _c is coarser of the two levels
            mesh_c = hierarchy[i-1]

            V_c = FunctionSpace(mesh_c, "Lagrange", 4)
            u_c = TrialFunction(V_c)
            v_c = TestFunction(V_c) 
            
            x_c, y_c = SpatialCoordinate(mesh_c)
            f_base_c = exp(-(((x_c-0.5)**2)/2) - (((y_c-0.5)**2)/2))
            a_c = (dot(grad(v_c), grad(u_c)) + v_c * u_c) * dx
            bcs_c = DirichletBC(V_c, 0, (1,2,3,4))

        sub_solutions = []
        # By this point function/ function spaces have been set up
        # Sampling now begins
        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))
            
            rand_multiplier = Constant(mult*rg.random_sample())
            # Element of randomness incorperated
            f_f = rand_multiplier*f_base_f
            L_f = f_f * v_f * dx 
            uh_f = Function(V_f)
            solve(a_f == L_f, uh_f, bcs=bcs_f, solver_parameters={'ksp_type': 'cg'})

        
            if i-1 >= 0:  
                f_c = rand_multiplier*f_base_c
                L_c = f_c * v_c * dx 
                uh_c = Function(V_c)
                solve(a_c == L_c, uh_c, bcs=bcs_c, solver_parameters={'ksp_type': 'cg'})
                
                uh_c2 = Function(V_f)
                prolong(uh_c, uh_c2)
                
                sub_solutions.append(uh_f - uh_c2)
            else:
                sub_solutions.append(uh_f)
        
        # This sum corresponds to the inner sum in the MLMC eqn.
        # This and prolong() is expensive when you have many repititions
        level_result = sum(sub_solutions)/Constant(repititions[i])
        
        if i != (levels - 1):
            # Interpolate to turn back into a function to allow prolong()
            level_result = interpolate(level_result, V_f)
            #temp = Function(V_high)
            #prolong(level_result, temp)
            #level_result = temp

        solutions.append(level_result)
    

    for i in range(len(solutions)-1):
        temp = Function(V_high)
        prolong(solutions[i], temp)
        solutions[i] = temp

    # Outer sum in MLMC eqn.
    end = time.time()
    print("Runtime: ", end - start, "s")
    estimate = sum(solutions)
    
    # Generate a ground truth result
    f = Constant(mult/2)*f_base_f
    L = f*v_f*dx
    u_real = Function(V_high)
    solve(a_f == L, u_real, bcs=bcs_f, solver_parameters={'ksp_type': 'cg'})

    difference = assemble(estimate - u_real)
    fig, axes = plt.subplots()
    collection = tripcolor(difference, axes=axes, cmap='coolwarm')
    fig.colorbar(collection)

    plt.show()
    # errornorm(u_real, interpolate(estimate, V_high))
    return errornorm(u_real, interpolate(estimate, V_high))

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

    return uh

def MLMC_general(coarse_fspace, levels, repititions, samples, isEval=True):
    """
    arg: coarse_fspace - FunctionSpace object on coarsest mesh
         levels - number of levels in MLMC
         repititions (list) - repititions at each level starting at coarsest
         samples (list) - list of all samples
         isEval (bool) - whether or not evaluation should be run on result
    output: Estimate of value
    """
    start = time.time()

    assert len(repititions) == levels, \
    ("The levels arguement is not equal to the number of entries in the iterable")

    assert len(samples) == sum(repititions), \
    ("Number of samples and sum of repetitions do not match.")

    solutions = P_sums()

    ######## ----- HOW CAN I ABSTRACT AWAY FROM THIS ------------
    coarse_mesh = coarse_fspace.mesh()
    family = coarse_fspace.ufl_element().family()
    degree = coarse_fspace.ufl_element().degree()
    hierarchy = MeshHierarchy(coarse_mesh, levels-1, 1)
    ######### ---------------------------------------------------

    sample_i = 0
    # Iterate through each level in hierarchy
    for i in range(levels):
        print("LEVEL {} - {} Samples".format(i+1, repititions[i]))
        
        sub_solutions = P_sums()

        # By this point function/ function spaces have been set up
        # Sampling now begins
        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))
            
            term = P_term(samples[sample_i], hierarchy, i, family, degree)
            term.calculate() # Calculate term value with ith sample
            sub_solutions.add_term(term)
            
            sample_i += 1  # move to next sample
        
        # This sum corresponds to the inner sum in the MLMC eqn.
        # This and prolong() is expensive when you have many repititions
        # interpolate happens here too to make sum/ division a function again for prolong()
        level_result = sub_solutions.average_terms()
        
        solutions.add_term(level_result)
    
    
    # Outer sum in MLMC eqn.
    estimate = solutions.sum_terms()
    end = time.time()
    print("Runtime: ", end - start, "s")

    if isEval:
        eval_soln(estimate.get_value(), Constant(10), hierarchy[-1])

    return estimate


def MLMC_general_field(levels, repititions, samples, isEval=True):
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
    assert len(samples) == sum(repititions), \
    ("Number of samples and sum of repetitions do not match.")

    solver = MLMC_Solver(levels)

    sample_i = 0
    # Iterate through each level in hierarchy
    for i in range(levels):
        print("LEVEL {} - {} Samples".format(i+1, repititions[i]))

        # By this point function/ function spaces have been set up
        # Sampling now begins
        for j in range(repititions[i]):
            print("Sample {} of {}".format(j+1, repititions[i]))
            
            solver.addTerm(samples[sample_i], i)

            sample_i += 1  # move to next sample
        
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
    
    # Creating random samples
    rg = RandomGenerator(MT19937(12345))
    samples = [Constant(20*rg.random_sample()) for i in range(sum(repititions))]
    
    # Creating base coarse mesh and function space
    coarse_mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(coarse_mesh, "Lagrange", 4)

    #estimate = MLMC_general(V, levels, repititions, samples, True)
    estimate = MLMC_general_field(levels, repititions, samples, True)




class MLMC_Solver:
    def __init__(self, levels):
        s = time.time()
        self._hierarchy = MeshHierarchy(UnitSquareMesh(10, 10), levels-1, 1)
        print("hierarchy time: ", time.time() - s)
        self._family = "Lagrange"
        self._degree = 4

        self._solutions = P_sums()
        self._sub_solutions = P_sums()

        self._result = None

    def initialiseInnerSum(self):
        self._sub_solutions = P_sums()
    
    def addTerm(self, sample, level):
        term = P_term(sample, self._hierarchy, level, self._family, self._degree)
        term.calculate()
        self._sub_solutions.add_term(term)
    
    def calculateInnerSum(self):
        level_result = self._sub_solutions.average_terms()
        self._solutions.add_term(level_result)
        self.initialiseInnerSum()

    def calculateOuterSum(self):
        self._result = self._solutions.sum_terms().get_value()
        return self._result
    
    def eval_result(self):
        
        uh_true = prob(self._hierarchy[-1], Constant(10))

        difference = assemble(self._result - uh_true)

        fig, axes = plt.subplots()
        collection = tripcolor(difference, axes=axes, cmap='coolwarm')
        fig.colorbar(collection)
        plt.show()

        print(errornorm(self._result, uh_true))


class P_sums:
    def __init__(self):
        self.__terms = []
    
    def add_term(self, p_term):
        self.__terms.append(p_term)
    
    def average_terms(self):
        result = sum(self.__terms)/Constant(len(self.__terms))
        result.interpolate_sum()
        return result
    
    def sum_terms(self):
        # do all prolonging at end
        self.sanitise()
        ans = sum(self.__terms)
        return ans
    
    def sanitise(self):
        fine_term = self.__terms[0]
        V_high = fine_term.fine_space()
        for term in self.__terms:
            if term._level+1 != len(term._hierarchy):
                temp = Function(V_high)
                prolong(term._value, temp)
                term._value = temp



class P_term:
    
    def __init__(self, sample, hierarchy, level, family, degree):
        self._sample = sample
        self._level = level
        self._hierarchy = hierarchy
        self._value = None
        self._family = family
        self._degree = degree
    
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

    def interpolate_sum(self):
        self._value = interpolate(self._value, self.current_space())
    
    def fine_space(self):
        fine_mesh = self._hierarchy[-1]
        return FunctionSpace(fine_mesh, self._family, self._degree)
     
    def current_space(self):
        fine_mesh = self._hierarchy[self._level]
        return FunctionSpace(fine_mesh, self._family, self._degree)
    
    # Should this be an input?
    def problem(self, mesh):
        V = FunctionSpace(mesh, "Lagrange", 4)
        u = TrialFunction(V)
        v = TestFunction(V)
        x, y = SpatialCoordinate(mesh)
        base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
        a = (dot(grad(v), grad(u)) + v * u) * dx
        bcs = DirichletBC(V, 0, (1,2,3,4))

        f = self._sample*base_f
        L = f * v * dx
        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
        return uh

    def calculate(self):
        assert self._value == None, ("Error: Already calculated")

        mesh_f = self._hierarchy[self._level]
        V_f = FunctionSpace(mesh_f, self._family, self._degree)
        # Call problem fuction
        uh_f = self.problem(mesh_f)
    
        if self._level-1 >= 0:  
            mesh_c = self._hierarchy[self._level-1]
            uh_c = self.problem(mesh_c)
                
            uh_c2 = Function(V_f)
            prolong(uh_c, uh_c2)
            
            self._value =  uh_f - uh_c2
        else:
            self._value = uh_f
        return 0
            
        
       


    
    

if __name__ == '__main__':
    #print(MC(40, 50, 20))
    #print(MLMC([10,20,40], [50,10,5], 20))
    #print(test1())
    #print(MLMC_hier(10, 3, [50,10, 5], 20))
    #print(MLMC_general(10, 3, [50,10, 5], 20, [1]))
    #general_test()