"""Verification of the normal-slip velocity BC (backend/slip.py) — multiphase step 1.

`solve_overdamped_slip` solves `-ν∇²v + γv = f` with a Nitsche BC that pins only the
**normal** velocity `v·n = g` and leaves the tangential traction free (perfect slip) —
the fluid-vs-membrane interface condition the multiphase model needs. The checks:

1. **Consistency / exactness** — a manufactured polynomial solution (whose normal-slip
   data is built from the *discrete* facet normal) is recovered to round-off; the Nitsche
   consistency terms make it accurate, not merely penalty-limited.
2. **The constraint is enforced** — `v·n → g` on the boundary, tightening with the penalty.
3. **Slip vs no-slip — the physics** — a tangential forcing drives a boundary flow that
   *slips* (`v_t ≠ 0`) under the normal-slip BC, where a no-slip (full-Dirichlet) BC kills
   it. This is what makes "normal pinned, tangential free" a real, distinguishable BC.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import make_disk_geometry, solve_overdamped_slip


def _disk(h: float = 0.04):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


def _boundary_normal_tangential(vh: fem.Function) -> tuple[float, float]:
    """(max |v·n|, mean |v_t|) over the boundary nodes, using the radial normal."""
    coords = vh.function_space.tabulate_dof_coordinates()[:, :2]
    r = np.linalg.norm(coords, axis=1)
    on_boundary = r > 0.98
    v = vh.x.array.reshape(-1, 2)[on_boundary]
    normals = coords[on_boundary] / r[on_boundary, None]
    v_normal = (v * normals).sum(axis=1)
    v_tangential = v - v_normal[:, None] * normals
    return float(np.abs(v_normal).max()), float(np.linalg.norm(v_tangential, axis=1).mean())


@pytest.mark.parametrize("exact", [(1.0, 0.0), (0.0, 1.0)])
def test_recovers_a_constant_field_to_roundoff(exact: tuple[float, float]) -> None:
    # v_exact = const ⇒ f = γ·v_exact, g = v_exact·n. Built with the discrete normal so the
    # penalty is geometry-consistent; the constant is in the P1 space, so exact recovery.
    mesh = _disk()
    const = ufl.as_vector([exact[0], exact[1]])
    n = ufl.FacetNormal(mesh)
    vh = solve_overdamped_slip(mesh, forcing=const, normal_velocity=ufl.dot(const, n), viscosity=1.0, screening=1.0)
    assert np.abs(vh.x.array.reshape(-1, 2) - np.array(exact)).max() < 1e-10


def test_recovers_a_linear_field_to_roundoff() -> None:
    # v_exact = x (position): -ν∇²x = 0, so f = γ·x; v·n = x·n. Linear ⇒ exact in P1, and
    # its tangential traction vanishes (∂_n x = n), so free-slip is satisfied.
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    vh = solve_overdamped_slip(mesh, forcing=x, normal_velocity=ufl.dot(x, n), viscosity=1.0, screening=1.0)
    nodes = vh.function_space.tabulate_dof_coordinates()[:, :2]
    assert np.abs(vh.x.array.reshape(-1, 2) - nodes).max() < 1e-10


def _weak_normal_residual(vh: fem.Function) -> float:
    # sqrt(∫_Γ (v·n)² ds) with the *discrete* facet normal — the residual the BC actually
    # controls (a nodal v·n with the smooth normal floors at the O(h) geometric mismatch).
    mesh = vh.function_space.mesh
    n = ufl.FacetNormal(mesh)
    form = fem.form(ufl.dot(vh, n) ** 2 * ufl.ds(domain=mesh))
    return math.sqrt(float(fem.assemble_scalar(form).real))


def test_constraint_tightens_with_penalty() -> None:
    # No penetration (g = 0) under a rotational forcing: the weak normal residual shrinks
    # as the Nitsche penalty β grows (the constraint is enforced weakly).
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    forcing = ufl.as_vector([-x[1], x[0]])  # wants to flow tangentially
    small = _weak_normal_residual(solve_overdamped_slip(mesh, forcing=forcing, normal_velocity=0.0 * x[0], beta=10.0))
    large = _weak_normal_residual(solve_overdamped_slip(mesh, forcing=forcing, normal_velocity=0.0 * x[0], beta=1000.0))
    assert large < small  # stronger penalty ⇒ tighter no-penetration
    assert large < 2e-3  # ≪ the ~0.18 tangential boundary flow it coexists with


def test_nonsymmetric_penalty_free_variant_matches_symmetric() -> None:
    # The non-symmetric Nitsche is coercive without a penalty, so it runs penalty-free
    # (beta=0, no parameter to tune) — robust for cut/embedded interfaces and weak
    # coercivity later. On these drag-coercive problems it must agree with the symmetric
    # default: recover a manufactured field exactly and discriminate slip identically.
    mesh = _disk()
    const = ufl.as_vector([1.0, 0.0])
    n = ufl.FacetNormal(mesh)
    vh = solve_overdamped_slip(mesh, forcing=const, normal_velocity=ufl.dot(const, n), symmetric=False, beta=0.0)
    assert np.abs(vh.x.array.reshape(-1, 2) - np.array([1.0, 0.0])).max() < 1e-10  # still exact

    x = ufl.SpatialCoordinate(mesh)
    forcing = ufl.as_vector([-x[1], x[0]])
    _, vt = _boundary_normal_tangential(
        solve_overdamped_slip(mesh, forcing=forcing, normal_velocity=0.0 * x[0], symmetric=False, beta=0.0)
    )
    assert vt > 0.05  # the fluid still slips, penalty-free


def test_slip_allows_tangential_flow_where_no_slip_forbids_it() -> None:
    # The physics check. A rotational forcing with no-penetration (v·n = 0): the slip BC
    # leaves a real tangential boundary flow; a no-slip BC (full Dirichlet v = 0, here a
    # huge normal+tangential penalty) suppresses it.
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    forcing = ufl.as_vector([-x[1], x[0]])

    slip = solve_overdamped_slip(mesh, forcing=forcing, normal_velocity=0.0 * x[0])
    _, vt_slip = _boundary_normal_tangential(slip)

    # A no-slip reference: full-vector Dirichlet v = 0 on the boundary.
    space = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u, w = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = (ufl.inner(ufl.grad(u), ufl.grad(w)) + ufl.inner(u, w)) * ufl.dx
    rhs = ufl.inner(forcing, w) * ufl.dx
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    from dolfinx.mesh import exterior_facet_indices

    bdofs = fem.locate_dofs_topological(space, mesh.topology.dim - 1, exterior_facet_indices(mesh.topology))
    noslip = fem.Function(space)
    from dolfinx.fem.petsc import LinearProblem

    LinearProblem(
        a,
        rhs,
        bcs=[fem.dirichletbc(fem.Function(space), bdofs)],
        u=noslip,
        petsc_options_prefix="test_noslip_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
    _, vt_noslip = _boundary_normal_tangential(noslip)

    assert vt_slip > 0.05  # the fluid genuinely slips along the boundary
    assert vt_noslip < 1e-9  # no-slip kills the tangential boundary velocity
    assert vt_slip > 1e6 * vt_noslip
