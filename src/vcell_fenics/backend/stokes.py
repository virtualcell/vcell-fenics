"""Incompressible Stokes (Taylor–Hood) — multiphase step 3, the first saddle point.

Step 3 of the multiphase-cytoplasm path (`docs/modeling/multiphase-cytoplasm-ale.md`)
replaces the screened vector-Laplacian stand-in with a genuine **incompressible Stokes**
phase: the symmetric-gradient viscous stress `2ν ε(v)` (ε = ½(∇v + ∇vᵀ)) plus a
**pressure** `p` enforcing `∇·v = 0`. The pressure is a Lagrange multiplier, so the
system is a **saddle point** — the first in the codebase — and needs inf-sup-stable
elements: **Taylor–Hood** (P2 velocity, P1 pressure).

Weak form (strong Dirichlet velocity for now; the Nitsche normal-slip BC with the *Stokes*
traction `(2ν ε(v) − p I)·n` is the next increment):

    ∫ 2ν ε(u):ε(v) dx − ∫ p ∇·v dx − ∫ q ∇·u dx = ∫ f·v dx

Under all-Dirichlet velocity the pressure is determined only up to a constant, so one
pressure dof is pinned to remove that null space (the field is physical up to the
constant). The saddle-point matrix is indefinite, so a direct LU with a pivoting solver
(MUMPS) is used rather than the plain LU that suffices for the positive-definite solves.

Verified (`tests/test_backend_stokes.py`): a manufactured solution — a div-free quadratic
velocity (exact in P2) and a linear pressure (exact in P1) — is recovered to round-off,
and `∇·u` is zero to round-off.
"""

from __future__ import annotations

import basix
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, exterior_facet_indices

from vcell_fenics.backend._typing import UflExpr


def solve_incompressible_stokes(
    mesh: Mesh,
    *,
    forcing: UflExpr,
    velocity: UflExpr,
    viscosity: float = 1.0,
) -> tuple[fem.Function, fem.Function]:
    """Solve incompressible Stokes `−∇·(2ν ε(u)) + ∇p = f`, `∇·u = 0`, with a strong
    Dirichlet velocity `u = velocity` on the whole boundary.

    `forcing` and `velocity` are UFL expressions on `mesh` (build with
    `ufl.SpatialCoordinate(mesh)`). Returns `(u, p)` as collapsed Taylor–Hood (P2/P1)
    `Function`s; the pressure is pinned at one dof (defined up to a constant otherwise).
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (2.0 * viscosity * ufl.inner(strain(u), strain(v)) - p * ufl.div(v) - q * ufl.div(u)) * dx
    rhs = ufl.inner(forcing, v) * dx

    # Strong Dirichlet velocity on the whole boundary.
    mesh.topology.create_connectivity(gdim - 1, gdim)
    facets = exterior_facet_indices(mesh.topology)
    velocity_space, _ = W.sub(0).collapse()
    bc_value = fem.Function(velocity_space)
    bc_value.interpolate(fem.Expression(velocity, velocity_space.element.interpolation_points))
    bc_velocity = fem.dirichletbc(
        bc_value, fem.locate_dofs_topological((W.sub(0), velocity_space), gdim - 1, facets), W.sub(0)
    )

    # Pin one pressure dof to 0 — the pressure is otherwise only determined up to a constant.
    _, pressure_to_mixed = W.sub(1).collapse()
    pin = np.array([pressure_to_mixed[0]], dtype=np.int32)
    bc_pressure = fem.dirichletbc(fem.Function(W), pin)

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        bcs=[bc_velocity, bc_pressure],
        u=solution,
        petsc_options_prefix=f"vcellfenics_stokes_{id(solution):x}_",
        # Saddle-point system is indefinite ⇒ a pivoting direct factorisation (MUMPS).
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()
