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

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import make_disk_geometry, solve_incompressible_stokes


def _disk(h: float = 0.06):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


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
    div_l2 = fem.assemble_scalar(fem.form(ufl.div(u) ** 2 * ufl.dx(domain=mesh)))
    assert float(mesh.comm.allreduce(div_l2, op=MPI.SUM)) ** 0.5 < 1e-9
