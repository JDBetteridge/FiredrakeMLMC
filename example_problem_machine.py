from firedrake import *

def problem(mesh, uh, alpha=1):
    V = FunctionSpace(mesh, "Lagrange", 4)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    base_f = exp(-(((x-0.5)**2)/2) - (((y-0.5)**2)/2))
    a = (dot(grad(v), grad(u)) + v * u) * dx

    bcs = DirichletBC(V, 0, (1,2,3,4))

    f = alpha*base_f
    L = f * v * dx
    return LinearVariationalProblem(a, L, uh, bcs=bcs)


if __name__ == '__main__':
    N = 10
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "Lagrange", 4)
    uh = Function(V)

    rg = RandomGenerator(MT19937(12345))

    lvp = problem(mesh, uh, alpha=1)
    lvs = LinearVariationalSolver(lvp, solver_parameters={'ksp_type': 'cg'})

    lvs.solve()

    #solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
