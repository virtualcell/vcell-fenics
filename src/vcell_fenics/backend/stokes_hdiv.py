"""Exactly mass-conserving incompressible Stokes on H(div) elements.

The Taylor–Hood Stokes (`backend/stokes.py`) is only *weakly* divergence-free, and on a
**moving** boundary the Nitsche normal-slip BC leaks ~3% divergence — fatal for a
conservative fluid–structure loop. This module fixes it at the root with an
**H(div)-conforming** discretization: a BDM velocity + DG pressure pair makes `∇·u = 0`
hold **pointwise** (the divergence of BDM_k lies in DG_{k-1}, so the constraint
`∫ q ∇·u = 0` forces it exactly), and the velocity normal component `v·n` is a *degree of
freedom*, so the slip BC `v·n = g` is imposed **strongly** — no Nitsche on the normal, no
pressure-test boundary term, no divergence pollution. The tangential traction is left free
(perfect slip), which for H(div) is simply *not* adding a boundary viscous term.

Because the velocity's tangential component is discontinuous across elements, the viscous
operator needs an **interior-penalty DG** treatment (consistency + symmetry + penalty on
the tangential jumps; the normal jump is zero by H(div)-conformity). A substrate-friction
`screening` makes the problem coercive (removing the rigid-mode null space of a pure
no-penetration BC, as in the CG path).

This is the "div-conforming element for the flow field" — the targeted near-term fix from
the DG discussion (`approaches.md`, "Discretization is a separate axis"), not the full DG
backend. The Nitsche-BC framework of the CG backend is bypassed *only* for the normal
velocity (now strong); everything else (the saddle-point MUMPS solve, the screening) carries
over.

Verified (`tests/test_backend_stokes_hdiv.py`): `∇·u` is zero to round-off on a *moving*
boundary (where Taylor–Hood leaked 3%), the fluid genuinely slips, and a manufactured
divergence-free solution is recovered.
"""

from __future__ import annotations

import basix
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, exterior_facet_indices

from vcell_fenics.backend._typing import UflExpr


def solve_incompressible_stokes_hdiv_slip(
    mesh: Mesh,
    *,
    boundary_velocity: fem.Function,
    forcing: UflExpr | None = None,
    viscosity: float = 1.0,
    screening: float = 0.0,
    degree: int = 2,
    penalty: float = 40.0,
) -> tuple[fem.Function, fem.Function]:
    """Solve incompressible Stokes with an exactly divergence-free H(div) discretization and
    a strong normal-slip BC `v·n = boundary_velocity·n` (free tangential).

    `boundary_velocity` is a vector `Function` whose *normal trace* on the boundary is the
    prescribed `v·n` (e.g. the mesh velocity `w` for a moving boundary, or a zero Function
    for no penetration) — only its normal component is used. `forcing` is an optional body
    force (UFL). `degree` is the BDM degree `k` (pressure is DG_{k-1}); `penalty` is the
    interior-penalty / DG stabilisation constant. Returns `(u, p)`: a **pointwise**
    divergence-free BDM velocity and its DG pressure.
    """

    velocity_element = basix.ufl.element("BDM", mesh.basix_cell(), degree)
    pressure_element = basix.ufl.element("DG", mesh.basix_cell(), degree - 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([velocity_element, pressure_element]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=mesh)
    dS = ufl.Measure("dS", domain=mesh)  # interior facets
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    h_avg = ufl.avg(h)
    nu = viscosity

    def jump(field: UflExpr) -> UflExpr:
        return field("+") - field("-")

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) + screening * ufl.inner(u, v) - p * ufl.div(v) - q * ufl.div(u)) * dx
    # Interior-penalty DG for the broken viscous operator (tangential jumps; normal jump = 0).
    a += (
        -ufl.inner(ufl.dot(ufl.avg(nu * ufl.grad(u)), n("+")), jump(v))
        - ufl.inner(ufl.dot(ufl.avg(nu * ufl.grad(v)), n("+")), jump(u))
        + (penalty * nu / h_avg) * ufl.inner(jump(u), jump(v))
    ) * dS

    zero = fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=np.float64))
    rhs = ufl.inner(forcing if forcing is not None else zero, v) * dx

    # Strong slip BC: v·n = boundary_velocity·n on the boundary normal dofs; tangential left free.
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = exterior_facet_indices(mesh.topology)
    velocity_space, _ = W.sub(0).collapse()
    boundary_in_space = fem.Function(velocity_space)
    boundary_in_space.interpolate(boundary_velocity)  # BDM preserves the normal trace
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), velocity_space), 1, facets)
    bc = fem.dirichletbc(boundary_in_space, boundary_dofs, W.sub(0))

    solution = fem.Function(W)
    LinearProblem(
        a,
        rhs,
        bcs=[bc],
        u=solution,
        petsc_options_prefix=f"vcellfenics_stokeshdiv_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    ).solve()
    return solution.sub(0).collapse(), solution.sub(1).collapse()
