"""Verification of the exactly-mass-conserving H(div) Stokes (backend/stokes_hdiv).

`solve_incompressible_stokes_hdiv_slip` solves incompressible Stokes on a BDM velocity +
DG pressure pair, with the normal-slip BC `v·n = g` imposed *strongly* on the H(div) normal
dofs. The point is exact pointwise mass conservation — including on a **moving** boundary,
where the Taylor–Hood Nitsche slip (`solve_incompressible_stokes_slip`) leaked ~3% `div`.

1. **Exact `div` on a moving boundary** — `g = cos(2θ) ≠ 0` (a bulging membrane): `∇·u` is
   zero to round-off and **does not grow with refinement** (the Taylor–Hood failure mode).
2. **It still slips** — a tangential boundary flow develops (the BC frees the tangent).
3. **No penetration** — with `g = 0` the boundary normal velocity is zero (strongly).
"""

from __future__ import annotations

import math

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import make_disk_geometry, solve_incompressible_stokes_hdiv_slip


def _disk(h: float):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


def _l2_div(u: fem.Function) -> float:
    mesh = u.function_space.mesh
    local = fem.assemble_scalar(fem.form(ufl.div(u) ** 2 * ufl.dx(domain=mesh)))
    return math.sqrt(float(mesh.comm.allreduce(local, op=MPI.SUM)))


def _bulging_boundary_velocity(mesh) -> fem.Function:  # type: ignore[no-untyped-def]
    # a vector field whose normal trace is cos(2θ): g·n with g = cos(2θ)·(x/r)
    space = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(ufl.dot(x, x))
    w = fem.Function(space)
    w.interpolate(fem.Expression(((x[0] ** 2 - x[1] ** 2) / (r * r)) * (x / r), space.element.interpolation_points))
    return w


def test_divergence_is_exact_on_a_moving_boundary() -> None:
    # cos(2θ) normal motion (a bulging membrane) — the case Taylor-Hood leaked ~3%. H(div)
    # with the strong normal BC keeps ∇·u at round-off, and it does NOT grow with refinement.
    divs = [
        _l2_div(
            solve_incompressible_stokes_hdiv_slip(
                m := _disk(h), boundary_velocity=_bulging_boundary_velocity(m), screening=1.0
            )[0]
        )
        for h in (0.08, 0.04, 0.02)
    ]
    assert max(divs) < 1e-8  # exact to round-off at every resolution


def test_fluid_slips_under_the_strong_normal_bc() -> None:
    # A rotational body force with no penetration (g = 0): the fluid develops a tangential
    # boundary flow (the tangent is free), with the normal velocity strongly zero.
    mesh = _disk(0.05)
    space = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    x = ufl.SpatialCoordinate(mesh)
    u, _ = solve_incompressible_stokes_hdiv_slip(
        mesh,
        boundary_velocity=fem.Function(space),  # g = 0 ⇒ no penetration
        forcing=ufl.as_vector([-x[1], x[0]]),
        screening=1.0,
    )
    assert _l2_div(u) < 1e-8  # still exactly divergence-free

    # BDM has moment dofs (no pointwise coordinates) — interpolate to Lagrange to inspect,
    # the same step visualization needs for H(div) fields.
    viz = fem.Function(space)
    viz.interpolate(u)
    coords = space.tabulate_dof_coordinates()[:, :2]
    r = np.linalg.norm(coords, axis=1)
    on_boundary = r > 0.98
    v = viz.x.array.reshape(-1, 2)[on_boundary]
    normals = coords[on_boundary] / r[on_boundary, None]
    v_normal = (v * normals).sum(axis=1)
    v_tangential = np.linalg.norm(v - v_normal[:, None] * normals, axis=1)
    # The strong BC zeros the BDM normal *moments*; the pointwise v·n (with the smooth
    # normal) is O(h) ≪ the tangential flow, not machine-zero.
    assert np.abs(v_normal).max() < 1e-2  # effectively no penetration
    assert v_tangential.mean() > 0.05  # genuine tangential slip
    assert v_tangential.mean() > 10 * np.abs(v_normal).max()  # tangential dominates
