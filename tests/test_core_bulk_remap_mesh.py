"""Verification of the DOLFINx bridge for the conservative bulk (2D area) remap.

Exercises the full pipeline — extracting (verts, tris, nodal values) off a P1
Function's mesh, the supermesh projection, and write-back into a Function on the
target mesh — against real DOLFINx meshes, mirroring the surface bridge's plan.

The kernel's exactness already lives in `test_core_bulk_remap.py`; here the job is
to prove the *bridge* (Function ↔ array, `V.dofmap.list` as the triangle list, the
P1 dof-index = vertex-row identity) is wired correctly, and that the conserve path
works on non-matching domains:

1. **Identity** — remapping onto the same space reproduces the field exactly.
2. **Constant preservation** — two triangulations of the *same* unit square (so the
   supermesh tiles exactly): a uniform field stays uniform. Any dof-ordering bug in
   the array extraction would break this.
3. **Linear preservation** — same-square, two resolutions: a linear field is
   reproduced exactly (the strong 2D exactness check).
4. **Mass conservation** — disk meshes at two resolutions (different polygons): with
   conserve=True the volume integral on the new mesh equals the old to round-off.
5. **Negative control** — naive nearest-vertex transfer drifts the volume integral
   where the conservative remap does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from vcell_fenics.approaches.submesh.geometry import create_disk_with_membrane
from vcell_fenics.core import remap_bulk_function


def _square_space(n: int) -> fem.FunctionSpace:
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, n, n, dmesh.CellType.triangle)
    return fem.functionspace(mesh, ("Lagrange", 1))


def _disk_space(h: float, *, radius: float = 1.0) -> fem.FunctionSpace:
    mesh = create_disk_with_membrane(radius=radius, h=h).bulk_mesh
    return fem.functionspace(mesh, ("Lagrange", 1))


def _set(V: fem.FunctionSpace, fn) -> fem.Function:  # type: ignore[no-untyped-def]
    u = fem.Function(V)
    u.interpolate(fn)
    return u


def _mass(u: fem.Function) -> float:
    return float(fem.assemble_scalar(fem.form(u * ufl.dx)).real)


# ---------------------------------------------------------------------------
# 1. identity
# ---------------------------------------------------------------------------


def test_identity_remap_reproduces_field() -> None:
    V = _disk_space(0.25)
    u = _set(V, lambda x: 1.0 + x[0] * x[0] + np.sin(x[1]))

    u_back = remap_bulk_function(u, V, conserve=False)

    assert np.allclose(u_back.x.array, u.x.array, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. constant preservation (same square, two resolutions → exact tiling)
# ---------------------------------------------------------------------------


def test_constant_preserved_across_resolutions() -> None:
    V_old, V_new = _square_space(5), _square_space(8)
    u_old = _set(V_old, lambda x: np.full(x.shape[1], 2.5))

    u_new = remap_bulk_function(u_old, V_new, conserve=False)

    assert np.allclose(u_new.x.array, 2.5, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. linear preservation (the strong 2D exactness check)
# ---------------------------------------------------------------------------


def test_linear_preserved_across_resolutions() -> None:
    V_old, V_new = _square_space(5), _square_space(8)
    u_old = _set(V_old, lambda x: 0.3 + 1.2 * x[0] - 0.7 * x[1])

    u_new = remap_bulk_function(u_old, V_new, conserve=False)

    xy = V_new.tabulate_dof_coordinates()
    exact = 0.3 + 1.2 * xy[:, 0] - 0.7 * xy[:, 1]
    assert np.allclose(u_new.x.array, exact, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. mass conservation across resolutions (disk: non-matching polygons)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("h_old", "h_new"), [(0.4, 0.18), (0.18, 0.4)])
def test_mass_conserved_across_resolutions(h_old: float, h_new: float) -> None:
    V_old, V_new = _disk_space(h_old), _disk_space(h_new)
    u_old = _set(V_old, lambda x: 1.5 + np.cos(2.0 * np.arctan2(x[1], x[0])))

    u_new = remap_bulk_function(u_old, V_new, conserve=True)

    assert _mass(u_new) == pytest.approx(_mass(u_old), rel=1e-11)


# ---------------------------------------------------------------------------
# 5. negative control — naive nearest-vertex transfer is not conservative
# ---------------------------------------------------------------------------


def test_nearest_vertex_copy_drifts_mass() -> None:
    V_old, V_new = _disk_space(0.4), _disk_space(0.18)
    u_old = _set(V_old, lambda x: 1.5 + np.cos(2.0 * np.arctan2(x[1], x[0])))

    # Naive transfer: each new dof takes the value of the geometrically nearest old
    # dof — accurate pointwise, but with no conservation guarantee.
    old_xyz = u_old.function_space.tabulate_dof_coordinates()
    new_xyz = V_new.tabulate_dof_coordinates()
    nearest = np.argmin(np.linalg.norm(new_xyz[:, None, :] - old_xyz[None, :, :], axis=2), axis=1)
    naive = fem.Function(V_new)
    naive.x.array[:] = u_old.x.array[nearest]

    conservative = remap_bulk_function(u_old, V_new, conserve=True)

    m_old = _mass(u_old)
    assert _mass(conservative) == pytest.approx(m_old, rel=1e-11)
    assert abs(_mass(naive) - m_old) / m_old > 1e-3  # the naive copy visibly drifts
