"""Incompressible Stokes (Taylor–Hood) — multiphase step 3, the first saddle point.

Step 3 of the multiphase-cytoplasm path (`docs/modeling/multiphase-cytoplasm-ale.md`)
replaces the screened vector-Laplacian stand-in with a genuine **incompressible Stokes**
phase: the symmetric-gradient viscous stress `2ν ε(v)` (ε = ½(∇v + ∇vᵀ)) plus a
**pressure** `p` enforcing `∇·v = 0`. The pressure is a Lagrange multiplier, so the
system is a **saddle point** — the first in the codebase — and needs inf-sup-stable
elements: **Taylor–Hood** (P2 velocity, P1 pressure).

Weak form (`solve_incompressible_stokes`, strong Dirichlet velocity):

    ∫ 2ν ε(u):ε(v) dx − ∫ p ∇·v dx − ∫ q ∇·u dx = ∫ f·v dx

`solve_incompressible_stokes_slip` instead applies the Nitsche normal-slip BC `u·n = g`
(free tangential) with the full **Stokes traction** `n·σ·n = 2ν n·ε(u)·n − p` — so the
pressure, and the pressure test, enter the boundary terms. A pure no-penetration BC on a
rotationally-symmetric domain leaves rigid rotation as a null mode, so that path takes an
optional substrate-friction `screening` to make the (screened) problem coercive.

`solve_incompressible_stokes_traction` applies a **traction** (Neumann) BC `σ·n = t` — the
membrane–cortex mechanical coupling (step 4): the membrane's surface mechanics load the
bulk fluid as a boundary traction. A tense membrane gives `t = −γ κ n`, and the fluid
returns the Laplace pressure `p = γ/R`.

`solve_incompressible_stokes_surface_tension` applies a uniform membrane **surface tension**
`γ` through the curvature-free weak load `−γ ∮_Γ ∇_Γ·v ds = −γ ∮ inner(P, ∇v) ds` (with the
surface projector `P = I − n⊗n`) — the force-balance FSI closure (`backend/fsi.py`), where
the membrane moves under its *own* tension. No explicit curvature: the Laplace–Beltrami /
continuous-surface-force identity moves the derivative onto the test function, so it works
on a piecewise-linear boundary where the pointwise curvature is undefined.

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


def solve_incompressible_stokes_slip(
    mesh: Mesh,
    *,
    forcing: UflExpr,
    normal_velocity: UflExpr,
    viscosity: float = 1.0,
    screening: float = 0.0,
    beta: float = 20.0,
    symmetric: bool = True,
) -> tuple[fem.Function, fem.Function]:
    """Incompressible Stokes with a Nitsche **normal-slip** BC `u·n = normal_velocity`,
    free tangential traction (perfect slip) — the membrane interface condition for a Stokes
    phase.

    Unlike the vector-Laplacian slip (`backend/slip.py`), the Nitsche consistency uses the
    full **Stokes traction** `n·σ·n = 2ν n·ε(u)·n − p`, so the pressure enters the boundary
    terms. `screening` (a substrate friction `γ`) makes the screened problem coercive and
    removes the rigid-mode null space — a pure no-penetration BC on a rotationally-symmetric
    domain otherwise leaves rigid rotation as a null mode. `symmetric` selects the Nitsche
    variant as in `nitsche_normal_slip`. Returns `(u, p)` (Taylor–Hood P2/P1); the pressure
    is pinned at one dof.
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    theta = -1.0 if symmetric else 1.0

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (
        2.0 * viscosity * ufl.inner(strain(u), strain(v))
        + screening * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * dx
    rhs = ufl.inner(forcing, v) * dx

    # Nitsche normal-slip with the Stokes traction n·σ·n = 2ν n·ε·n − p (so p, and the
    # pressure test q, enter the boundary terms); the tangential traction is left free.
    traction_u = 2.0 * viscosity * ufl.dot(ufl.dot(strain(u), n), n) - p
    traction_v = 2.0 * viscosity * ufl.dot(ufl.dot(strain(v), n), n) - q
    u_n, v_n = ufl.dot(u, n), ufl.dot(v, n)
    penalty = beta * 2.0 * viscosity / h
    a += (-traction_u * v_n + theta * traction_v * u_n + penalty * u_n * v_n) * ds
    rhs += (theta * traction_v * normal_velocity + penalty * normal_velocity * v_n) * ds

    # Pin one pressure dof — the normal-velocity BC does not fix the pressure constant.
    _, pressure_to_mixed = W.sub(1).collapse()
    bc_pressure = fem.dirichletbc(fem.Function(W), np.array([pressure_to_mixed[0]], dtype=np.int32))

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        bcs=[bc_pressure],
        u=solution,
        petsc_options_prefix=f"vcellfenics_stokesslip_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()


def solve_incompressible_stokes_traction(
    mesh: Mesh,
    *,
    traction: UflExpr,
    forcing: UflExpr | None = None,
    viscosity: float = 1.0,
    screening: float = 1.0,
) -> tuple[fem.Function, fem.Function]:
    """Incompressible Stokes with a **traction** (Neumann) boundary condition `σ·n =
    traction` — the membrane–cortex mechanical coupling (multiphase step 4).

    The membrane's surface mechanics exert a force on the enclosed fluid; that force is the
    boundary traction `σ·n` on the bulk Stokes phase. For a tense membrane the traction is
    the curvature force `−γ κ n` (inward), and at equilibrium the fluid responds with the
    Laplace pressure `p = γ κ = γ/R`. A traction BC is the *natural* BC of the Stokes weak
    form, so it enters only the right-hand side (`∮ traction·v ds`) — no Nitsche needed; the
    traction also fixes the pressure level (no constant null space to pin). A substrate
    friction `screening` removes the rigid-body velocity null space.

    `traction` (and the optional body `forcing`) are UFL expressions on `mesh` — build the
    curvature traction with `ufl.FacetNormal(mesh)`. Returns `(u, p)` (Taylor–Hood P2/P1).
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (
        2.0 * viscosity * ufl.inner(strain(u), strain(v))
        + screening * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * dx
    rhs = ufl.inner(traction, v) * ds  # natural BC: σ·n = traction
    if forcing is not None:
        rhs += ufl.inner(forcing, v) * dx

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        u=solution,
        petsc_options_prefix=f"vcellfenics_stokestraction_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()


def solve_incompressible_stokes_surface_tension(
    mesh: Mesh,
    *,
    tension: float | UflExpr,
    viscosity: float = 1.0,
    screening: float = 1.0,
) -> tuple[fem.Function, fem.Function]:
    """Incompressible Stokes driven by a membrane **surface tension** `γ`, for the force-balance
    FSI closure (the membrane moves under its own tension + the bulk pressure). `tension` is a
    uniform scalar *or* a **field** (a boundary-valued `Function`/expression) for a spatially
    varying tension — the mechano-chemical case where γ depends on a surface species (e.g.
    `γ = γ₀ + α·R` for a receptor density R).

    The tension enters as the weak boundary load `−∮_Γ γ ∇_Γ·v ds` — the Laplace–Beltrami /
    continuous-surface-force form, so **no explicit curvature** is computed (the surface
    divergence of the test velocity *is* the curvature force, integrated by parts). With γ inside
    the integral this automatically carries BOTH the normal curvature force `γκn` AND the
    tangential **Marangoni** force `∇_Γγ` from tension gradients — so a non-uniform tension drives
    a net flow (the surface contracts harder where γ is larger). At equilibrium a circle of
    uniform γ gives the Laplace pressure `p = γ/R` with `v ≈ 0`; a tension gradient breaks that
    symmetry and moves the cell.

    Taylor–Hood (P2/P1), so the velocity is **continuous** — moving the ALE mesh by it
    conserves area, since the constant-pressure mode enforces `∮ v·n = 0` exactly (an H(div)
    velocity, being discontinuous, loses that when interpolated for the mesh motion). A
    substrate-friction `screening` removes the rigid-body null space; the tension fixes the
    pressure level, so no pin is needed. Returns `(u, p)`.
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)
    n = ufl.FacetNormal(mesh)

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (
        2.0 * viscosity * ufl.inner(strain(u), strain(v))
        + screening * ufl.inner(u, v)
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * dx
    surface_projection = ufl.Identity(gdim) - ufl.outer(n, n)  # P = I − n⊗n
    rhs = -tension * ufl.inner(surface_projection, ufl.grad(v)) * ds  # −γ ∮ ∇_Γ·v ds

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        u=solution,
        petsc_options_prefix=f"vcellfenics_stokestension_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()
