from dune.fem.scheme import galerkin
import matplotlib.pyplot as plt
from ufl import TrialFunction, TestFunction, dx, inner, grad, pi
import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC
from numpy.random import normal



alpha = Constant(1., f"alpha_0")
f = Constant(1., f"f_0")
dim = 1
N   = 100
L = 1
grid = structuredGrid([0]*dim, [L]*dim, [N]*dim)

spatialOrder = 1
spatialBasis = lagrange(grid, spatialOrder)
U = TrialFunction(spatialBasis)
V = TestFunction(spatialBasis)
equation = 0.
K = inner( alpha * grad(U), grad(V) )* dx
F = -inner(f, V) * dx
equation = K + F

bc = DirichletBC(spatialBasis, [0.], None)

scheme = galerkin([equation == 0., bc], solver = ("istl", "gmres"), parameters = {"nonlinear.tolerance":1e-14, "nonlinear.errormeasure": "residualreduction"})
u_h = spatialBasis.function(name="u_h")
varianceVector = u_h.copy(name="v_h")
samples = 5
uhList = [u_h.copy(name=f"u_{i}") for i in range(1,samples)] + [u_h]
meanVector = u_h.copy(name="meanu")
for trialVector in uhList:
    f.value = 1. + 0.1*normal(0., 1.)
    alpha.value = 1. + 0.1*normal(0., 1.)
    scheme.solve(trialVector)

for solutions in uhList:
    meanVector.as_numpy[:] += solutions.as_numpy[:]
meanVector.as_numpy[:] /=samples
for i in range(len(uhList)):
    solutions = uhList[i]
    plt.plot(np.linspace(0, 1., N+1), solutions.as_numpy, label = f"$u_{i+1}$")
plt.plot(np.linspace(0, 1., N+1), meanVector.as_numpy, label = f"$E[u]$")
plt.legend()
plt.title("Stochastic modes")
plt.xlabel("x")
plt.ylabel("u")
plt.show()
for solutions in uhList:
    varianceVector.as_numpy[:] += (solutions.as_numpy[:] - meanVector.as_numpy[:]) ** 2
varianceVector.as_numpy[:] /= ( samples - 1)
plt.plot(np.linspace(0, 1., N+1), meanVector.as_numpy, label = f"$E[u]$")
plt.plot(np.linspace(0, 1., N+1), meanVector.as_numpy + 1.98 / np.sqrt(samples) * np.sqrt(varianceVector.as_numpy), "r--", label = "Confidence interval")
plt.plot(np.linspace(0, 1., N+1), meanVector.as_numpy - 1.98 / np.sqrt(samples) * np.sqrt(varianceVector.as_numpy), "r--")
plt.legend()
plt.show()
print(np.max(meanVector.as_numpy))
print(np.max(np.sqrt(varianceVector.as_numpy)))
