"""Verification of the Approach-A trace correction (core/surface_remap_trace).

The claim (docs/modeling/conservative-surface-remap.md, "The Approach-A-specific
wrinkle"): a conservative *bulk* remap does not conserve a boundary trace's
*surface* mass ∫_Γ ρ ds, so the trace needs a separate correction. These tests
build bulk disk meshes, treat ρ as the boundary trace of a bulk P1 function, and
check:

1. **The bug is real** — a naive boundary transfer drifts ∫_Γ ρ ds.
2. **The correction fixes it** — after correct_surface_trace, ∫_Γnew ρ ds equals
   ∫_Γold ρ ds to round-off.
3. **Interior untouched** — only boundary DOFs change.
4. **Surface measures agree** — ∫_Γ ρ ds on the bulk facet measure equals the
   submesh integral, so the correction's submesh conservation transfers to the trace.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh

from vcell_fenics.approaches.submesh.geometry import create_disk_with_membrane
from vcell_fenics.core import BulkBoundaryTrace, correct_surface_trace


def _bulk_space(h: float) -> fem.FunctionSpace:
    return fem.functionspace(create_disk_with_membrane(radius=1.0, h=h).bulk_mesh, ("Lagrange", 1))


def _trace_field(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.5 + np.cos(2.0 * np.arctan2(x[1], x[0])))


def _surface_mass(u: fem.Function) -> float:
    """∫_∂Ω u ds — the surface mass of the boundary trace, on the bulk facet measure."""
    return float(fem.assemble_scalar(fem.form(u * ufl.ds)).real)


def _boundary_dofs(V: fem.FunctionSpace) -> np.ndarray:
    bulk = V.mesh
    tdim = bulk.topology.dim
    bulk.topology.create_connectivity(tdim - 1, tdim)
    facets = dmesh.exterior_facet_indices(bulk.topology)
    return fem.locate_dofs_topological(V, tdim - 1, facets)


def _naive_boundary_copy(u_old: fem.Function, V_new: fem.FunctionSpace) -> fem.Function:
    """Stand-in for a bulk remap's boundary result: each new boundary dof takes the
    nearest old boundary dof's value — accurate pointwise, not conservative."""
    u_new = fem.Function(V_new)
    old_b = _boundary_dofs(u_old.function_space)
    new_b = _boundary_dofs(V_new)
    old_xy = u_old.function_space.tabulate_dof_coordinates()[old_b]
    new_xy = V_new.tabulate_dof_coordinates()[new_b]
    nearest = np.argmin(np.linalg.norm(new_xy[:, None, :] - old_xy[None, :, :], axis=2), axis=1)
    u_new.x.array[new_b] = u_old.x.array[old_b][nearest]
    return u_new


# ---------------------------------------------------------------------------
# 4. surface measures agree (the premise the correction relies on)
# ---------------------------------------------------------------------------


def test_bulk_facet_mass_matches_submesh() -> None:
    V = _bulk_space(0.3)
    u = fem.Function(V)
    u.interpolate(_trace_field)
    trace = BulkBoundaryTrace(V)
    rho = trace.gather(u)
    submesh_mass = float(fem.assemble_scalar(fem.form(rho * ufl.dx)).real)
    assert _surface_mass(u) == pytest.approx(submesh_mass, rel=1e-12)


# ---------------------------------------------------------------------------
# 1 + 2. the bug, and the correction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("h_old", "h_new"), [(0.4, 0.18), (0.18, 0.4)])
def test_correction_conserves_surface_mass(h_old: float, h_new: float) -> None:
    V_old, V_new = _bulk_space(h_old), _bulk_space(h_new)
    u_old = fem.Function(V_old)
    u_old.interpolate(_trace_field)

    u_new = _naive_boundary_copy(u_old, V_new)
    m_old = _surface_mass(u_old)

    # 1. The naive transfer drifts surface mass...
    assert abs(_surface_mass(u_new) - m_old) / m_old > 1e-3

    # 2. ...and the correction restores it exactly.
    correct_surface_trace(u_old, u_new, conserve=True)
    assert _surface_mass(u_new) == pytest.approx(m_old, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. interior DOFs are untouched
# ---------------------------------------------------------------------------


def test_correction_leaves_interior_untouched() -> None:
    V_old, V_new = _bulk_space(0.4), _bulk_space(0.18)
    u_old = fem.Function(V_old)
    u_old.interpolate(_trace_field)

    u_new = fem.Function(V_new)
    u_new.x.array[:] = 7.0  # sentinel everywhere
    interior = np.setdiff1d(np.arange(u_new.x.array.size), _boundary_dofs(V_new))

    correct_surface_trace(u_old, u_new)

    assert np.all(u_new.x.array[interior] == 7.0)  # interior sentinel intact
    assert not np.allclose(u_new.x.array[_boundary_dofs(V_new)], 7.0)  # boundary rewritten
