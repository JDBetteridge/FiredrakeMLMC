from firedrake import *
from mpi4py import MPI
from randomgen import RandomGenerator, MT19937

from mlmcparagen import MLMC_Solver, MLMC_Problem

rg = RandomGenerator(MT19937(12345))

class FiredrakeProblem(object):
    """

    """
    def __init__(self, functionspace):
        self._V = functionspace
        self._sample = Function(self._V)
        self._uh = Function(self._V)

        u = TrialFunction(self._V)
        v = TestFunction(self._V)

        a = (dot(grad(v), grad(u)) + v * u) * dx

        bcs = DirichletBC(self._V, 0, (1,2,3,4))
        f = self._sample
        L = f * v * dx
        problem = LinearVariationalProblem(a, L, self._uh, bcs=bcs)
        self._solver = LinearVariationalSolver(vp, solver_parameters={'ksp_type': 'cg'})

    def solve(self, sample):
        self._sample.assign(sample)
        self._solver.solve()
        return assemble(Constant(0.5) * dot(self._uh, self._uh) * dx)

def sampler(V_f, V_c):
    rand = 20*rg.random_sample()
    sample_c = None

    x_f, y_f = SpatialCoordinate(V_f.mesh())
    base_f = exp(-(((x_f-0.5)**2)/2) - (((y_f-0.5)**2)/2))
    sample_f = Function(V_f)
    sample_f.interpolate(Constant(rand)*base_f)

    if V_c is not None:
        x_c, y_c = SpatialCoordinate(V_c.mesh())
        base_c = exp(-(((x_c-0.5)**2)/2) - (((y_c-0.5)**2)/2))
        sample_c = Function(V_c)
        sample_c.interpolate(Constant(rand)*base_c)

    return sample_f, sample_c

def level_maker(finelevel, coarselevel, comm=MPI.COMM_WORLD):
    coarse_mesh = UnitSquareMesh(10, 10)
    hierarchy = MeshHierarchy(coarse_mesh, finelevel, 1)
    V_f = FunctionSpace(hierarchy[finelevel], "CG", 2)
    if coarselevel < 0:
        return V_f, None
    else:
        V_c = FunctionSpace(hierarchy[coarselevel], "CG", 2)
        return V_f, V_c

# Levels and repetitions
levels = 3
repetitions = [100, 50, 10]
MLMCprob = MLMC_Problem(FiredrakeProblem, sampler, level_maker)
MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
estimate = MLMCsolv.solve()

print(estimate)
