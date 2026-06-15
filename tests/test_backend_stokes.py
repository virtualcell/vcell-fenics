"""Verification of incompressible Stokes (Taylor–Hood) — multiphase step 3.

`solve_incompressible_stokes` solves `−∇·(2ν ε(u)) + ∇p = f`, `∇·u = 0`, with a strong
Dirichlet velocity — the codebase's first saddle-point system (a pressure Lagrange
multiplier, inf-sup-stable P2/P1 elements). The check is a manufactured solution:

- velocity `u = (2xy, −x² − y²)` — divergence-free, quadratic ⇒ exact in P2,
- pressure `p = x` — linear ⇒ exact in P1,
- so `f = −ν∇²u + ∇p = (1, 4ν)`.

Both are recovered to round-off (the manufactured fields lie in the Taylor–Hood space),
and `∇·u` is zero to round-off — verifying the symmetric-gradient operator, the pressure /
incompressibility constraint, and the saddle-point solve together.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend import (
    make_disk_geometry,
    solve_incompressible_stokes,
    solve_incompressible_stokes_slip,
)


def _disk(h: float = 0.06):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


def _l2_div(u: fem.Function) -> float:
    mesh = u.function_space.mesh
    local = fem.assemble_scalar(fem.form(ufl.div(u) ** 2 * ufl.dx(domain=mesh)))
    return math.sqrt(float(mesh.comm.allreduce(local, op=MPI.SUM)))


def _boundary_normal_tangential(u: fem.Function) -> tuple[float, float]:
    coords = u.function_space.tabulate_dof_coordinates()[:, :2]
    r = np.linalg.norm(coords, axis=1)
    b = r > 0.98
    v = u.x.array.reshape(-1, 2)[b]
    nrm = coords[b] / r[b, None]
    vn = (v * nrm).sum(axis=1)
    return float(np.abs(vn).max()), float(np.linalg.norm(v - vn[:, None] * nrm, axis=1).mean())


@pytest.mark.parametrize("nu", [1.0, 0.25])
def test_recovers_a_manufactured_stokes_solution(nu: float) -> None:
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    u_exact = ufl.as_vector([2 * x[0] * x[1], -(x[0] ** 2) - x[1] ** 2])
    forcing = ufl.as_vector([1.0, 4.0 * nu])

    u, p = solve_incompressible_stokes(mesh, forcing=forcing, velocity=u_exact, viscosity=nu)

    # Velocity: exact in P2.
    coords = u.function_space.tabulate_dof_coordinates()[:, :2]
    u_exact_nodal = np.column_stack([2 * coords[:, 0] * coords[:, 1], -(coords[:, 0] ** 2) - coords[:, 1] ** 2])
    assert np.abs(u.x.array.reshape(-1, 2) - u_exact_nodal).max() < 1e-9

    # Pressure: linear, exact in P1, up to the pinned constant ⇒ compare zero-mean.
    p_coords = p.function_space.tabulate_dof_coordinates()[:, 0]
    p_error = (p.x.array - p.x.array.mean()) - (p_coords - p_coords.mean())
    assert np.abs(p_error).max() < 1e-9

    # Incompressibility.
    assert _l2_div(u) < 1e-9


# ---------------------------------------------------------------------------
# Step 3b — the Nitsche normal-slip BC with the Stokes traction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
def test_slip_recovers_a_slip_compatible_manufactured_solution(symmetric: bool) -> None:
    # A slip-compatible manufactured solution: constant u=[1,0] (zero strain ⇒ tangential
    # traction = −p·n_t = 0, free-slip satisfied), linear p=x, so f = ∇p = (1,0); the slip
    # data g = u·n is built from the discrete normal. Recovered to round-off — verifying the
    # Stokes-traction Nitsche terms (pressure in the boundary terms) for both variants.
    mesh = _disk()
    n = ufl.FacetNormal(mesh)
    e1 = ufl.as_vector([1.0, 0.0])
    u, p = solve_incompressible_stokes_slip(
        mesh, forcing=e1, normal_velocity=ufl.dot(e1, n), symmetric=symmetric, beta=20.0
    )
    assert np.abs(u.x.array.reshape(-1, 2) - np.array([1.0, 0.0])).max() < 1e-9

    p_coords = p.function_space.tabulate_dof_coordinates()[:, 0]
    p_error = (p.x.array - p.x.array.mean()) - (p_coords - p_coords.mean())
    assert np.abs(p_error).max() < 1e-9
    assert _l2_div(u) < 1e-9


def test_slip_is_incompressible_on_a_well_posed_flow() -> None:
    # A null-mode-orthogonal forcing f=[1,0] with substrate friction: a well-posed slip flow
    # is divergence-free to round-off. (A forcing aligned with the rigid-rotation null mode —
    # which no-penetration leaves unconstrained — would instead be inconsistent.)
    mesh = _disk(h=0.04)
    x = ufl.SpatialCoordinate(mesh)
    u, _ = solve_incompressible_stokes_slip(
        mesh, forcing=ufl.as_vector([1.0, 0.0]), normal_velocity=0.0 * x[0], screening=1.0
    )
    assert _l2_div(u) < 1e-9


def test_slip_allows_tangential_flow_where_no_slip_forbids_it() -> None:
    # Rotational forcing with substrate friction (which damps the rigid-rotation null mode):
    # the slip BC leaves a tangential boundary flow with no penetration, where a no-slip
    # (full-Dirichlet) Stokes solve kills it.
    mesh = _disk(h=0.05)
    x = ufl.SpatialCoordinate(mesh)
    rotation = ufl.as_vector([-x[1], x[0]])

    u_slip, _ = solve_incompressible_stokes_slip(mesh, forcing=rotation, normal_velocity=0.0 * x[0], screening=1.0)
    # A no-slip reference: the Dirichlet Stokes solve with u = 0 on the boundary (a zero
    # fem.Constant, not a UFL `0` that would drop its integration domain).
    zero_velocity = fem.Constant(mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
    u_noslip, _ = solve_incompressible_stokes(mesh, forcing=rotation, velocity=zero_velocity, viscosity=1.0)

    vn_slip, vt_slip = _boundary_normal_tangential(u_slip)
    _, vt_noslip = _boundary_normal_tangential(u_noslip)
    assert vn_slip < 1e-2  # no penetration
    assert vt_slip > 0.1  # the fluid slips along the boundary
    assert vt_slip > 50 * vt_noslip  # where no-slip kills the tangential flow
