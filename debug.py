from firedrake import *

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
        self._sample.assign(Constant(sample))
        self._vs.solve()
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

mesh0 = UnitSquareMesh(20,20)
mesh1 = UnitSquareMesh(40,40)
V0 = FunctionSpace(mesh0, "CG", 2)
V1 = FunctionSpace(mesh1, "CG", 2)
l0 = problemClass(V0)
l1 = problemClass(V1)

rg = RandomGenerator(MT19937(12345))
ans = [20*rg.random_sample() for i2 in range(2)]

print(l0.solve(ans[1]))
print(l1.solve(ans[1]), l0.solve(ans[1]))