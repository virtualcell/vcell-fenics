"""Overdamped fluid phase on a *moving* mesh with a normal-matching slip BC — the
integration that completes multiphase step 1 (`docs/modeling/multiphase-cytoplasm-ale.md`).

It composes three pieces already verified in isolation: the Nitsche normal-slip BC
(`backend/slip.py`), `relative_advection` (the ALE convective term), and the mesh motion.
Each step solves the fluid velocity `v` with the slip BC `v·n = w·n` (the fluid follows the
moving boundary normally, slips tangentially), advances a scalar species by
`relative_advection = v − w`, and moves the mesh by `w`. The checks:

1. **Still fluid follows the boundary** — on a rotating mesh a still (forcing-free) fluid
   solves to ≈0 (the boundary's true normal motion is zero); the small residual is the
   O(h) facet-leakage of the discrete normal (a known geometric limitation), and it
   shrinks under refinement.
2. **Slip in the moving context** — a driven fluid develops a real tangential boundary
   flow while `v·n` stays pinned to `w·n` (no penetration).
3. **Velocity-solved Eulerian discriminator** — feeding the *solved* `v − w` into
   `relative_advection`, a lab-frame field stays put while the mesh rotates through it
   (the PR-#8 discriminator, now with the velocity solved not prescribed), to the O(h)
   geometric floor.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend import (
    BackwardEuler,
    DiscreteProblem,
    Term,
    TermKind,
    make_disk_geometry,
    solve_overdamped_slip,
)

_OMEGA = 0.6


def _disk(h: float):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


def _rotation(mesh, space: fem.FunctionSpace) -> fem.Function:  # type: ignore[no-untyped-def]
    """The rigid-rotation mesh velocity w = ω(−y, x) as a P1 vector Function."""
    x = ufl.SpatialCoordinate(mesh)
    w = fem.Function(space)
    w.interpolate(fem.Expression(ufl.as_vector([-_OMEGA * x[1], _OMEGA * x[0]]), space.element.interpolation_points))
    return w


def _boundary_normal_tangential(vh: fem.Function) -> tuple[float, float]:
    coords = vh.function_space.tabulate_dof_coordinates()[:, :2]
    r = np.linalg.norm(coords, axis=1)
    b = r > 0.98
    v = vh.x.array.reshape(-1, 2)[b]
    nn = coords[b] / r[b, None]
    vn = (v * nn).sum(axis=1)
    return float(np.abs(vn).max()), float(np.linalg.norm(v - vn[:, None] * nn, axis=1).mean())


def test_still_fluid_follows_the_moving_boundary_to_geometric_floor() -> None:
    # No forcing + slip v·n = w·n on a rotating mesh ⇒ the fluid is at rest, up to the O(h)
    # facet-leakage of the discrete w·n (chords aren't tangent). It shrinks with h.
    n_coarse = _max_still_velocity(0.06)
    n_fine = _max_still_velocity(0.03)
    assert n_fine < n_coarse  # the residual is geometric — refining the mesh reduces it
    assert n_fine < 0.05


def _max_still_velocity(h: float) -> float:
    mesh = _disk(h)
    space = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    w = _rotation(mesh, space)
    v = solve_overdamped_slip(mesh, forcing=fem.Function(space), normal_velocity=ufl.dot(w, ufl.FacetNormal(mesh)))
    return float(np.abs(v.x.array).max())


def test_driven_fluid_slips_on_the_moving_mesh() -> None:
    # A tangential body force with v·n = w·n: the fluid develops a real tangential boundary
    # flow (slips) while the no-penetration constraint holds.
    mesh = _disk(0.05)
    space = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    w = _rotation(mesh, space)
    x = ufl.SpatialCoordinate(mesh)
    v = solve_overdamped_slip(
        mesh, forcing=ufl.as_vector([-x[1], x[0]]), normal_velocity=ufl.dot(w, ufl.FacetNormal(mesh))
    )
    vn, vt = _boundary_normal_tangential(v)
    assert vt > 0.05  # genuine tangential slip
    assert vn < 5e-3  # but no penetration (matches w·n ≈ 0)


def _velocity_solved_discriminator_error(h: float, *, dt: float = 0.01, n_steps: int = 40) -> float:
    """Run the coupled loop and return max|c − lab_field|: a lab-frame field should stay
    put while the mesh rotates through it, with the velocity SOLVED and v−w driving
    `relative_advection`."""
    mesh = _disk(h)
    space_v = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    space_c = fem.functionspace(mesh, ("Lagrange", 1))
    normal = ufl.FacetNormal(mesh)

    rel = fem.Function(space_v)  # relative_advection = v − w, read by the species term
    w = fem.Function(space_v)
    c_trial, c_test = ufl.TrialFunction(space_c), ufl.TestFunction(space_c)
    species = DiscreteProblem(
        variable_name="c",
        V=space_c,
        trial=c_trial,
        test=c_test,
        dx=ufl.Measure("dx", domain=mesh),
        unknown=fem.Function(space_c, name="c"),
        previous=fem.Function(space_c, name="c_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(dt)),  # type: ignore[operator]
        terms=(Term(TermKind.TIME_DERIVATIVE), Term(TermKind.ADVECTION, ufl.dot(rel, ufl.grad(c_trial)) * c_test)),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=None,
    )
    species.interpolate_initial(1.0 + 0.3 * ufl.SpatialCoordinate(mesh)[0])
    geom_from_dof = cKDTree(space_v.tabulate_dof_coordinates()).query(mesh.geometry.x)[1]

    for _ in range(n_steps):
        w.interpolate(_rotation(mesh, space_v))
        v = solve_overdamped_slip(mesh, forcing=fem.Function(space_v), normal_velocity=ufl.dot(w, normal))
        rel.x.array[:] = v.x.array - w.x.array
        species.step()
        mesh.geometry.x[:, :2] += (dt * w.x.array).reshape(-1, 2)[geom_from_dof]

    coords = space_c.tabulate_dof_coordinates()[:, :2]
    lab_field = 1.0 + 0.3 * coords[:, 0]
    return float(np.abs(species.unknown.x.array - lab_field).max())


def test_velocity_solved_eulerian_discriminator() -> None:
    # The integration: slip-solved v feeds relative_advection = v − w, and the lab field is
    # held static while the mesh rotates. Accuracy is the O(h) geometric floor, so it halves
    # under refinement — distinguishing a real solve from a co-moving (field-rotates) bug.
    coarse = _velocity_solved_discriminator_error(0.05)
    fine = _velocity_solved_discriminator_error(0.025)
    assert coarse < 1e-2  # the lab field stays put (a co-moving bug would give ~0.07)
    assert fine < 0.7 * coarse  # error is geometric — refining the mesh reduces it
