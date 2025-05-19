from dune.fem.scheme import galerkin
import matplotlib.pyplot as plt
from ufl import TrialFunction, TestFunction, dx, inner, grad, pi
import chaospy as pce
import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC
dist = pce.Normal(0.,1.)
orderPCE = 4
PCEBasis = pce.orth_ttr(orderPCE, dist)
M = orderPCE + 1

omegaFactor = pce.E(PCEBasis[:, None, None]* PCEBasis[None, :, None] * PCEBasis[None, None, :], dist)
alpha_1 = Constant(0.1, f"alpha_1")
alpha = [Constant(1., f"alpha_0")] + [alpha_1] * (M - 1)
f_1 = Constant(0.1, f"f_1")
f = [Constant(1., f"f_0")] + [f_1] * (M - 1)
assert len(alpha) <= M and len(f) <= M
dim = 1
N   = 100
L = 1
grid = structuredGrid([0]*dim, [L]*dim, [N]*dim)

spatialOrder = 1
spatialBasis = lagrange(grid, spatialOrder, dimRange = M)
U = TrialFunction(spatialBasis)
V = TestFunction(spatialBasis)
equation = 0.
for k in range(M):
    for j in range(M):
        for i in range(len(alpha)):
            E_ijk = omegaFactor[i,j,k]
            if abs(E_ijk) > 1e-14:
                K_i = inner(alpha[i] * grad(U[j]), grad(V[k]) )* dx
                equation += E_ijk * K_i
    if k < len(f):
        equation -= inner(f[k], V[k]) * dx


bc = DirichletBC(spatialBasis, [0.] * M, None)

scheme = galerkin([equation == 0., bc], solver = ("istl", "gmres"), parameters = {"linear.verbose": True, "nonlinear.tolerance":1e-14, "nonlinear.errormeasure": "residualreduction"})
soloBasis = lagrange(grid, 1, 1)
u_h = spatialBasis.function(name="u_h")
scheme.solve(u_h)
for i in range(M):
    plt.plot(np.linspace(0, 1., N+1), u_h.as_numpy[i::M], label = f"$u_{i}$")
plt.legend()
plt.title("Stochastic modes")
plt.xlabel("x")
plt.ylabel("u")
plt.show()
varianceVector = soloBasis.function(name="Vuh")
E_psi_squared = [pce.E((psi - pce.E(psi, dist))**2, dist) for psi in PCEBasis]
for i in range(1,M):
    varianceVector.as_numpy[:] += u_h.as_numpy[i::M]**2 * E_psi_squared[i]

plt.plot(np.linspace(0, 1., N+1), u_h.as_numpy[::M], label = f"$u_0$")
plt.plot(np.linspace(0, 1., N+1), u_h.as_numpy[::M] + 1.98 * np.sqrt(varianceVector.as_numpy), "r--", label = "Confidence interval")
plt.plot(np.linspace(0, 1., N+1), u_h.as_numpy[::M] - 1.98 * np.sqrt(varianceVector.as_numpy), "r--")
plt.legend()
plt.show()
print(np.max(u_h.as_numpy[::M]))
print(np.max(np.sqrt(varianceVector.as_numpy)))
