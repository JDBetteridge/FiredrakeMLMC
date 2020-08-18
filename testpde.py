
import matplotlib.pyplot as plt
from firedrake import *

print(1+1)


mesh1 = UnitSquareMesh(40, 40)
mesh2 = UnitSquareMesh(40, 40)
print(mesh1.size(0))
print(mesh2.size(1))
"""
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.legend();
"""
"""
V = FunctionSpace(mesh, "Lagrange", 4)
u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
f = (1 + 8*pi*pi)*cos(2*pi*x)*cos(2*pi*y)

n = FacetNormal(mesh)
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx 

bcs = DirichletBC(V, 0, (1,2,3,4))

uh = Function(V)

solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg'})

print(assemble(dot(uh, uh) * dx))
"""
"""
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.show()
"""
