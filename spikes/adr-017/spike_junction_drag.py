import numpy as np, ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
eta, lam, zeta, gamma, mu = 1.0, 1.0, 1.0, 1.0, 1.0
for N in (8, 32, 128, 512):
    m = dmesh.create_rectangle(MPI.COMM_WORLD, [[-1, -1], [1, 1]], [N, N], dmesh.CellType.triangle)
    fac = dmesh.locate_entities(m, 1, lambda x: np.isclose(x[1], 0) & (x[0] > -1e-12))
    ft = dmesh.meshtags(m, 1, np.sort(fac).astype(np.int32), np.ones(len(fac), dtype=np.int32))
    V = fem.functionspace(m, ("Lagrange", 1, (2,)))
    u, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))
    a = fem.form((2*eta*ufl.inner(eps(u), eps(phi)) + lam*ufl.div(u)*ufl.div(phi) + zeta*ufl.dot(u, phi))*ufl.dx)
    n = ufl.FacetNormal(m); g = ufl.grad(phi)("+"); nn = n("+")
    L = fem.form(-gamma*(ufl.tr(g) - ufl.dot(nn, ufl.dot(g, nn)))*ufl.Measure("dS", domain=m, subdomain_data=ft)(1))
    bfac = dmesh.locate_entities_boundary(m, 1, lambda x: np.isclose(np.abs(x[0]), 1) | np.isclose(np.abs(x[1]), 1))
    bc = fem.dirichletbc(np.zeros(2), fem.locate_dofs_topological(V, 1, bfac), V)
    A = assemble_matrix(a, bcs=[bc]); A.assemble()
    X = V.tabulate_dof_coordinates()[:, :2]; i0 = int(np.argmin(np.linalg.norm(X, axis=1)))
    for c in range(2): A.setValue(2*i0+c, 2*i0+c, mu, addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()
    b = assemble_vector(L); apply_lifting(b, [a], [[bc]]); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); set_bc(b, [bc])
    x = b.duplicate(); ksp = PETSc.KSP().create(); ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.solve(b, x)
    print(f"h=2/{N:4d}  v(origin)={x.array.reshape(-1,2)[i0]}  (F/mu = {gamma/mu})")
