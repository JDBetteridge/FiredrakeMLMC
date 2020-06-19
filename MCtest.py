import matplotlib.pyplot as plt
from firedrake import *
from randomgen import RandomGenerator, MT19937
import time

"""
rg = RandomGenerator(MT19937(12345))
for i in range(10):
    print(rg.random_sample())

"""
def MC(mesh_size, levels, mult):
    start = time.time()

    mesh = UnitSquareMesh(mesh_size, mesh_size)

    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()


    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    rg = RandomGenerator(MT19937(12345))
    solutions = []

    #levels = 50
    #mult = 20

    for i in range(levels):
        print("Sample {} of {}".format(i+1, levels))
        rand_multiplier = Constant(mult*rg.random_sample())
        f = rand_multiplier*base_f

        L = f * v * dx 

        uh = Function(V)

        solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
        solutions.append(uh)

    f = Constant(mult/2)*base_f
    L = f*v*dx
    u_real = Function(V)
    solve(a == L, u_real, bcs=bcs, solver_parameters={'ksp_type': 'cg'})

    estimate = sum(solutions)/Constant(levels)
    print(type(estimate))
    print(type(interpolate(estimate, V)))
    end = time.time()
    print(end-start)
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


def MLMC_hier(coarse_size, levels, repititions, mult):
    start = time.time()

    solutions = []
    rg = RandomGenerator(MT19937(12345))

    coarse_mesh = UnitSquareMesh(coarse_size, coarse_size)
    hierarchy = MeshHierarchy(coarse_mesh, levels-1, 1)

    V_high = FunctionSpace(hierarchy[-1], "Lagrange", 4)

    for i in range(len(hierarchy)):
        mesh_f = hierarchy[i]
        
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
            mesh_c = hierarchy[i-1]

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
                
                uh_c2 = Function(V_f)
                prolong(uh_c, uh_c2)
                
                sub_solutions.append(uh_f - uh_c2)
            else:
                sub_solutions.append(uh_f)
        
        level_result = sum(sub_solutions)/Constant(repititions[i])
        
        level_result = interpolate(level_result, V_f)
        
        
        if i != (levels - 1):
            temp = Function(V_high)
            prolong(level_result, temp)
            level_result = temp

        solutions.append(level_result)
    
    f = Constant(mult/2)*f_base_f
    L = f*v_f*dx
    u_real = Function(V_high)
    solve(a_f == L, u_real, bcs=bcs_f, solver_parameters={'ksp_type': 'cg'})

    estimate = sum(solutions)
    
    end = time.time()
    print(end - start)

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

    """
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    L = f * v * dx 
    """
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



if __name__ == '__main__':
    print(MC(40, 50, 20))
    #print(MLMC([10,20,40], [50,10,5], 20))
    #print(test1())
    #print(MLMC_hier(10, 3, [50,10, 5], 20))