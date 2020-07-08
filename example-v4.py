from firedrake import *
from randomgen import RandomGenerator, MT19937

from MLMCv4 import MLMC_general_scalar, MLMC_Solver, P_level, convergence_tests

rg = RandomGenerator(MT19937(12345))
def samp():
    return 20*rg.random_sample()

class problemClass:
    """
    Needs to take an integer initialisation argument to define the level (0 - L)
    Needs to have a .solve() method which takes a sample as an argument and returns
    a scalar solution
    """
    def __init__(self, level):
        
        self._V = self.level_maker(level)
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
    
    # HELPER
    def level_maker(self, level):
        coarse_mesh = UnitSquareMesh(10, 10)
        hierarchy = MeshHierarchy(coarse_mesh, level, 1)
        return FunctionSpace(hierarchy[-1], "Lagrange", 4)



def general_test():
    # Levels and repititions
    levels = 3
    repititions = [1000, 200, 10]
    estimate = MLMC_general_scalar(problemClass, samp, levels, repititions, True)


if __name__ == '__main__':
    general_test()
