"""Spike: an open tense segment ending at the origin inside a viscous, drag-screened bulk.
The weak tension form's endpoint term is a point force; does the endpoint velocity converge in h?"""
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
eta, lam, zeta, gamma = 1.0, 1.0, 1.0, 1.0
for N in (8, 16, 32, 64, 128, 256):
    m = dmesh.create_rectangle(MPI.COMM_WORLD, [[-1, -1], [1, 1]], [N, N], dmesh.CellType.triangle)
    m.topology.create_connectivity(1, 2)
    fac = dmesh.locate_entities(m, 1, lambda x: np.isclose(x[1], 0) & (x[0] > -1e-12))
    ft = dmesh.meshtags(m, 1, np.sort(fac).astype(np.int32), np.ones(len(fac), dtype=np.int32))
    V = fem.functionspace(m, ("Lagrange", 1, (2,)))
    u, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    a = (2 * eta * ufl.inner(eps(u), eps(phi)) + lam * ufl.div(u) * ufl.div(phi) + zeta * ufl.dot(u, phi)) * ufl.dx
    n = ufl.FacetNormal(m); g = ufl.grad(phi)("+"); nn = n("+")
    dS = ufl.Measure("dS", domain=m, subdomain_data=ft)
    L = -gamma * (ufl.tr(g) - ufl.dot(nn, ufl.dot(g, nn))) * dS(1)
    bfac = dmesh.locate_entities_boundary(m, 1, lambda x: np.isclose(np.abs(x[0]), 1) | np.isclose(np.abs(x[1]), 1))
    bc = fem.dirichletbc(np.zeros(2), fem.locate_dofs_topological(V, 1, bfac), V)
    vh = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="s").solve()
    X = V.tabulate_dof_coordinates()[:, :2]; i0 = np.argmin(np.linalg.norm(X, axis=1))
    i1 = np.argmin(np.linalg.norm(X - [0.5, 0.0], axis=1)); arr = vh.x.array.reshape(-1, 2)
    print(f"h=2/{N:4d}  v(origin)={arr[i0]}  v(0.5,0)={arr[i1]}")
