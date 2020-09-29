"""
L_1, L_0 = level_maker(1, 0) # Create level objects
P_0 = Problem_Class(L_0) # Initialise problem on level 0
P_1 = Problem_Class(L_1) # Initialise problem on level 1

Y = 0 # Level 1 result
N1 = 10 # Samples on level 1

# Generate and solve on N1 random samples
for n in range(N1):
    sample_1, sample_0 = sampler(L_1, L_0)
    Y += (P_1.solve(sample_1) - P_0.solve(sample_0))

Y /= N1

################################

levels = 3
repetitions = [100, 50, 25]
limits = [1, 2, 4]

MLMCprob = MLMC_Problem(Problem_Class, sampler, level_maker)
MLMCsolver_1 = MLMC_Solver(MLMCprob, levels, repetitions)
MLMCsolver_2 = MLMC_Solver(MLMCprob, levels, repetitions, 
comm=MPI.COMM_WORLD, comm_limits=limits)
"""
"""
from firedrake import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

uh = Function(V) # Where solution is held
u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
A = Function(V)
f = A*exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
a = (dot(grad(u), grad(v)) + u * v) * dx
L = f * v * dx

bcs = DirichletBC(V, 0, (1,2,3,4))
vp = LinearVariationalProblem(a, L, uh, bcs=bcs)
vs = LinearVariationalSolver(vp, solver_parameters={"ksp_type": "cg"})

results = []
rg = RandomGenerator(MT19937(12345))
samples = [Constant(20*rg.random_sample()) for i in range(100)]
for i in samples:
    A.assign(i)
    vs.solve()
    results.append(assemble(dot(uh, uh) * dx))

print(sum(results)/len(results))
"""
from firedrake import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10, 10)
hierarchy = MeshHierarchy(mesh, 1, 1)
V_f = FunctionSpace(hierarchy[-1], "CG", 4)
V_c = FunctionSpace(hierarchy[0], "CG", 4)

beta_c = Function(V_c)
rg = RandomGenerator(MT19937(12345))
beta_f = rg.beta(V_f, 1.0, 2.0)
inject(beta_f, beta_c)

fig, axes = plt.subplots()
collection = tripcolor(beta_c, axes=axes, cmap='coolwarm')
fig.colorbar(collection)

fig, axes = plt.subplots()
collection = tripcolor(beta_f, axes=axes, cmap='coolwarm')
fig.colorbar(collection)

plt.show()