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

Verified (`tests/test_backend_multiphase.py`): a manufactured two-field solution is
recovered to round-off (block + drag + slip assembly); increasing `ξ` **locks** the
phases (the slip `|v_n − v_s| → 0`); and at `ξ = 0` each phase reduces to the independent
single-phase slip solve (the drag is the only coupling).
"""

from __future__ import annotations

import basix
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
