from firedrake import *
import matplotlib.pyplot as plt

class problemClass:
    def __init__(self, V):
        
        self._V = V

        self._sample = Function(self._V)
        self._qh = Function(self._V)
        self._vs = self.initialise_problem()
    
    def solve(self, sample):
        
        self._qh.assign(0)
        self._sample.assign(sample)

        print(assemble(dot(self._sample, self._sample) * dx))

        self._vs.solve()

        result = assemble(dot(self._qh, self._qh) * dx)
        print(result)
        return result
        """
        qh = Function(self._V)
        q = TrialFunction(self._V)
        p = TestFunction(self._V)

        a = inner(exp(sample)*grad(q), grad(p))*dx
        L = inner(Constant(1.0), p)*dx
        bcs = DirichletBC(self._V, Constant(0.0), (1,2,3,4))

        vp = LinearVariationalProblem(a, L, qh, bcs=bcs)
        solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
        vs = LinearVariationalSolver(vp, solver_parameters=solver_param) 
        vs.solve()
        result = assemble(dot(qh, qh) * dx)
        print(result)
        return result
        """

    def initialise_problem(self):

        q = TrialFunction(self._V)
        p = TestFunction(self._V)

        a = inner(exp(self._sample)*grad(q), grad(p))*dx
        L = inner(Constant(1.0), p)*dx
        bcs = DirichletBC(self._V, Constant(0.0), (1,2,3,4))

        vp = LinearVariationalProblem(a, L, self._qh, bcs=bcs, constant_jacobian=False)
        solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
        
        return LinearVariationalSolver(vp, solver_parameters=solver_param)


rg = RandomGenerator(PCG64(12345))

mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, "CG", 2)

problem = problemClass(V)

for i in range(10):
    print("Sample {}".format(i))
    sample = rg.beta(V, 1, 3)
    problem.solve(sample)