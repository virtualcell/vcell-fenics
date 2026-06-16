"""Two-phase overdamped momentum balance with interphase drag — multiphase step 2.

The second machinery piece of the multiphase-cytoplasm path
(`docs/modeling/multiphase-cytoplasm-ale.md`, step 2): two inertia-free velocity fields
— a cytoskeleton *network* `v_n` and a *solvent* `v_s` — coupled by an **interphase
drag** `ξ(v_a − v_b)` and solved as one **block system**. This is what makes the model
genuinely two-phase (an active gel + its solvent) rather than two independent solves.

Each phase obeys an overdamped (no `∂_t v`) force balance

    -ν_a ∇²v_a + γ_a v_a + ξ (v_a − v_b) = f_a     (a, b = n, s)

— viscous + an optional substrate friction `γ_a` (e.g. focal-adhesion / cortex-membrane
drag, which also lifts the pure-viscous rigid-mode null space) + the symmetric interphase
drag `ξ(v_a − v_b)` (the off-diagonal coupling) = forcing `f_a` (active stress, pressure
gradient, …). Each phase carries its own **normal-slip BC** `v_a·n = g_a` (the membrane
interface condition, `backend/slip.py`), so a phase can slip tangentially while not
penetrating the boundary.

`-ν∇²` stands in for the viscous stress until the symmetric-gradient Stokes operator and
the incompressible-mixture pressure land (step 3, the first saddle-point system). The
block is assembled monolithically over a mixed element of two vector spaces and solved
directly.

`solve_two_phase_stokes` adds the symmetric-gradient stress and the **incompressible-mixture
pressure** (step 3c, the saddle point); `solve_two_phase_stokes_surface_tension` drives that
mixture with a **membrane surface tension** instead of prescribed slip — the bulk side of the
two-phase force-balance FSI closure (`backend/fsi.py`).

Verified (`tests/test_backend_multiphase.py`): a manufactured two-field solution is
recovered to round-off (block + drag + slip assembly); increasing `ξ` **locks** the
phases (the slip `|v_n − v_s| → 0`); and at `ξ = 0` each phase reduces to the independent
single-phase slip solve (the drag is the only coupling).
"""

from __future__ import annotations

import basix
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.slip import nitsche_normal_slip


def solve_two_phase_overdamped(
    mesh: Mesh,
    *,
    drag: float,
    forcing_n: UflExpr,
    forcing_s: UflExpr,
    normal_velocity_n: UflExpr,
    normal_velocity_s: UflExpr,
    viscosity_n: float = 1.0,
    viscosity_s: float = 1.0,
    screening_n: float = 0.0,
    screening_s: float = 0.0,
    beta: float = 20.0,
    symmetric: bool = True,
) -> tuple[fem.Function, fem.Function]:
    """Solve the coupled two-phase overdamped balance, returning `(v_n, v_s)`.

    `drag` is the interphase friction `ξ`; `forcing_*` and `normal_velocity_*` are UFL
    expressions on `mesh` (the per-phase body force and the slip-BC normal velocity
    `g_a = v_a·n`, e.g. `0` for no penetration or `ufl.dot(w_mesh, n)` to match a moving
    boundary). `screening_*` is the optional per-phase substrate friction (also removes
    the rigid-mode null space when no slip BC pins it). Returns the two phase velocities
    as collapsed P1 vector `Function`s.
    """

    gdim = mesh.geometry.dim
    element = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(gdim,))
    W = fem.functionspace(mesh, basix.ufl.mixed_element([element, element]))
    v_n, v_s = ufl.TrialFunctions(W)
    w_n, w_s = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)

    a = (
        viscosity_n * ufl.inner(ufl.grad(v_n), ufl.grad(w_n))
        + viscosity_s * ufl.inner(ufl.grad(v_s), ufl.grad(w_s))
        + screening_n * ufl.inner(v_n, w_n)
        + screening_s * ufl.inner(v_s, w_s)
        + drag * ufl.inner(v_n - v_s, w_n)  # interphase drag — the off-diagonal coupling
        + drag * ufl.inner(v_s - v_n, w_s)
    ) * dx
    rhs = (ufl.inner(forcing_n, w_n) + ufl.inner(forcing_s, w_s)) * dx

    a_n, rhs_n = nitsche_normal_slip(
        v_n, w_n, normal_velocity_n, mesh=mesh, ds=ds, viscosity=viscosity_n, beta=beta, symmetric=symmetric
    )
    a_s, rhs_s = nitsche_normal_slip(
        v_s, w_s, normal_velocity_s, mesh=mesh, ds=ds, viscosity=viscosity_s, beta=beta, symmetric=symmetric
    )

    solution = fem.Function(W, name="two_phase_velocity")
    LinearProblem(
        a + a_n + a_s,
        rhs + rhs_n + rhs_s,
        u=solution,
        petsc_options_prefix=f"vcellfenics_twophase_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()


def solve_two_phase_stokes(
    mesh: Mesh,
    *,
    drag: float,
    forcing_n: UflExpr,
    forcing_s: UflExpr,
    normal_velocity_n: UflExpr,
    normal_velocity_s: UflExpr,
    viscosity_n: float = 1.0,
    viscosity_s: float = 1.0,
    screening_n: float = 0.0,
    screening_s: float = 0.0,
    beta: float = 20.0,
    symmetric: bool = True,
) -> tuple[fem.Function, fem.Function, fem.Function]:
    """The two-phase **incompressible** mixture — multiphase step 3c, the culmination of the
    momentum machinery: the interphase-drag block of `solve_two_phase_overdamped` with the
    Stokes pressure of `solve_incompressible_stokes`.

    Two velocity fields `(v_n, v_s)` and one **mixture pressure** `p` enforcing
    `∇·(v_n + v_s) = 0` (the simplest equal-volume-fraction incompressible mixture). Each
    phase has the symmetric-gradient viscous stress, optional substrate friction, the
    symmetric interphase drag `ξ(v_a − v_b)`, and its own Nitsche normal-slip BC whose
    Stokes traction `2ν n·ε(v_a)·n − p` carries the **shared** pressure. Assembled over a
    Taylor–Hood mixed element `[P2, P2, P1]` and solved with a pivoting (MUMPS) direct solve.

    Returns `(v_n, v_s, p)`; the pressure is pinned at one dof. As with the single-phase
    slip, a `screening` removes the rigid-rotation null mode of a pure no-penetration BC.

    Verified (`tests/test_backend_multiphase.py`): a manufactured mixture (`v_n = [1,0]`,
    `v_s = [−1,0]` so the sum is divergence-free; linear pressure) recovered to round-off,
    and `∇·(v_n + v_s)` zero to round-off on a generic well-posed flow.
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p2, p1]))
    v_n, v_s, p = ufl.TrialFunctions(W)
    w_n, w_s, q = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    theta = -1.0 if symmetric else 1.0

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (
        2.0 * viscosity_n * ufl.inner(strain(v_n), strain(w_n))
        + 2.0 * viscosity_s * ufl.inner(strain(v_s), strain(w_s))
        + screening_n * ufl.inner(v_n, w_n)
        + screening_s * ufl.inner(v_s, w_s)
        + drag * ufl.inner(v_n - v_s, w_n)
        + drag * ufl.inner(v_s - v_n, w_s)
        - p * ufl.div(w_n + w_s)  # one mixture pressure, conjugate to the total velocity
        - q * ufl.div(v_n + v_s)  # incompressible mixture
    ) * dx
    rhs = (ufl.inner(forcing_n, w_n) + ufl.inner(forcing_s, w_s)) * dx

    # Per-phase Nitsche normal-slip; the Stokes traction carries the *shared* mixture pressure.
    penalty = beta * 2.0 / h
    for velocity, test, g, nu in (
        (v_n, w_n, normal_velocity_n, viscosity_n),
        (v_s, w_s, normal_velocity_s, viscosity_s),
    ):
        traction_v = 2.0 * nu * ufl.dot(ufl.dot(strain(velocity), n), n) - p
        traction_w = 2.0 * nu * ufl.dot(ufl.dot(strain(test), n), n) - q
        v_dot_n, w_dot_n = ufl.dot(velocity, n), ufl.dot(test, n)
        a += (-traction_v * w_dot_n + theta * traction_w * v_dot_n + nu * penalty * v_dot_n * w_dot_n) * ds
        rhs += (theta * traction_w * g + nu * penalty * g * w_dot_n) * ds

    _, pressure_to_mixed = W.sub(2).collapse()
    bc_pressure = fem.dirichletbc(fem.Function(W), np.array([pressure_to_mixed[0]], dtype=np.int32))

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        bcs=[bc_pressure],
        u=solution,
        petsc_options_prefix=f"vcellfenics_twophasestokes_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse(), solution.sub(2).collapse()


def solve_two_phase_stokes_surface_tension(
    mesh: Mesh,
    *,
    tension: float,
    drag: float,
    viscosity_n: float = 1.0,
    viscosity_s: float = 1.0,
    screening_n: float = 1.0,
    screening_s: float = 1.0,
) -> tuple[fem.Function, fem.Function, fem.Function]:
    """The two-phase incompressible mixture driven by a **membrane surface tension** `γ`, for
    the two-phase force-balance FSI closure (`backend/fsi.py`): the membrane moves under its
    own tension while the bulk is the network + solvent mixture of `solve_two_phase_stokes`.

    Like `solve_incompressible_stokes_surface_tension` but for the two-phase mixture: the
    tension enters as the curvature-free weak load `−γ ∮_Γ ∇_Γ·v ds`, split by **equal volume
    fraction** (½ each) over the two phase test velocities so the *total* boundary load is the
    single Laplace traction — at a circle this gives the mixture pressure `p = γ/R` with both
    phases at rest, and the drag `ξ(v_n − v_s)` inactive. No slip BC (the membrane is free,
    the natural BC); a substrate `screening_*` removes each phase's rigid-body null space and
    the tension fixes the pressure level (no pin needed). Returns `(v_n, v_s, p)`.
    """

    gdim = mesh.geometry.dim
    p2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([p2, p2, p1]))
    v_n, v_s, p = ufl.TrialFunctions(W)
    w_n, w_s, q = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.ds(domain=mesh)
    n = ufl.FacetNormal(mesh)

    def strain(field: UflExpr) -> UflExpr:
        return ufl.sym(ufl.grad(field))

    a = (
        2.0 * viscosity_n * ufl.inner(strain(v_n), strain(w_n))
        + 2.0 * viscosity_s * ufl.inner(strain(v_s), strain(w_s))
        + screening_n * ufl.inner(v_n, w_n)
        + screening_s * ufl.inner(v_s, w_s)
        + drag * ufl.inner(v_n - v_s, w_n)
        + drag * ufl.inner(v_s - v_n, w_s)
        - p * ufl.div(w_n + w_s)  # one mixture pressure, conjugate to the total velocity
        - q * ufl.div(v_n + v_s)  # incompressible mixture
    ) * dx
    surface_projection = ufl.Identity(gdim) - ufl.outer(n, n)  # P = I − n⊗n
    # Tension splits ½/½ over the phases ⇒ the total boundary load is the single Laplace traction.
    tension_load = ufl.inner(surface_projection, ufl.grad(w_n)) + ufl.inner(surface_projection, ufl.grad(w_s))
    rhs = -0.5 * tension * tension_load * ds

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        u=solution,
        petsc_options_prefix=f"vcellfenics_twophasetension_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse(), solution.sub(2).collapse()
