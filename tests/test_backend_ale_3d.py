"""3D moving meshes (the ALE path on tetrahedra): prescribed boundary motion, the harmonic interior
extension, the conservative time term and lab-frame transport — the same machinery as 2D, which had
only ever run on triangles."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_DT = 0.05


def _sphere_problem(velocity: str, terms: str, initial: str) -> DiscreteProblem:
    geometry = GeometryDescription(
        name="ball",
        dim=3,
        extent=(4.0, 4.0, 4.0),
        origin=(-2.0, -2.0, -2.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < 1.0"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )
    model = f"""
math_description:
  geometry: ball
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
    - {{ name: ext, kind: volume }}
  variables:
    - {{ name: u, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: u
      subdomain: cyto
      temporality: time_dependent
      terms: {{ {terms} }}
      initial_condition: "{initial}"
"""
    problem = assemble(load_yaml(model), realize(geometry, h=0.25), dt=_DT)
    assert isinstance(problem, DiscreteProblem)
    return problem


def _integral(problem: DiscreteProblem, integrand: object) -> float:
    return float(fem.assemble_scalar(fem.form(integrand * ufl.dx(domain=problem.V.mesh))).real)


def _advance(problem: DiscreteProblem, t_final: float) -> None:
    for k in range(1, round(t_final / _DT) + 1):
        problem.set_time(k * _DT)
        problem.step()


def test_a_translating_sphere_keeps_its_volume_and_moves_exactly() -> None:
    problem = _sphere_problem("[0.4, -0.2, 0.3]", 'diffusion: "0.1", advection: "[0.0, 0.0, 0.0]"', "1.0 + geom.x[0]")
    x = ufl.SpatialCoordinate(problem.V.mesh)
    volume0, mass0 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    centroid0 = np.array([_integral(problem, x[i]) for i in range(3)]) / volume0
    _advance(problem, 1.0)
    volume1, mass1 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    centroid1 = np.array([_integral(problem, x[i]) for i in range(3)]) / volume1
    assert volume1 == pytest.approx(volume0, rel=1e-12)  # a rigid motion: the tetrahedra are only moved
    assert np.allclose(centroid1 - centroid0, [0.4, -0.2, 0.3], atol=1e-12)
    assert mass1 == pytest.approx(mass0, rel=1e-12)  # lab-frame species swept by the front: the total is kept


def test_a_deforming_sphere_keeps_its_mass() -> None:
    # an axisymmetric squeeze at the equator (the 3D furrow's shape of motion)
    ring = "-0.5 * exp(-geom.x[1]**2 / 0.1)"
    velocity = f"[{ring} * geom.x[0], 0.0, {ring} * geom.x[2]]"
    problem = _sphere_problem(velocity, 'diffusion: "0.1", advection: "[0.0, 0.0, 0.0]"', "1.0 + geom.x[1]")
    volume0, mass0 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    _advance(problem, 0.5)
    assert _integral(problem, problem.unknown) == pytest.approx(mass0, rel=1e-12)
    assert _integral(problem, 1.0) < volume0  # the equator moved inward


# -- 3D remeshing (backend/remesh_3d.py) and the 3D transfer (core/bulk_remap_mesh.py) ---------------------


def _ball_mesh(
    expression: str = "geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < 1.0", h: float = 0.25, extent: float = 4.0
) -> object:
    geometry = GeometryDescription(
        name="ball",
        dim=3,
        extent=(extent, extent, extent),
        origin=(-extent / 2, -extent / 2, -extent / 2),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression=expression),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )
    return realize(geometry, h=h).mesh_of("cyto")


def _volume_of(mesh: object) -> float:
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh))).real)


def test_the_3d_remesh_keeps_the_region() -> None:
    from vcell_fenics.backend.remesh_3d import outward_boundary, remesh_region_3d

    old = _ball_mesh()
    points, triangles = outward_boundary(old)  # type: ignore[arg-type]
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    enclosed = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    assert enclosed == pytest.approx(_volume_of(old), rel=1e-12)  # outward: the divergence theorem closes
    new = remesh_region_3d(old, 0.25)  # type: ignore[arg-type]
    assert _volume_of(new) == pytest.approx(_volume_of(old), rel=1e-12)  # the volume is restored exactly
    corners = np.asarray(new.geometry.x, dtype=np.float64)[np.asarray(new.geometry.dofmap)]
    signed = np.einsum(
        "ij,ij->i",
        corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
    )
    assert np.all(np.abs(signed) > 0.0)  # no degenerate tetrahedra
    # the new boundary lies on the old one (the unit sphere, meshed at h = 0.25)
    radius = np.linalg.norm(outward_boundary(new)[0], axis=1)
    assert radius.min() > 0.95 and radius.max() < 1.03


def test_the_3d_transfer_is_exact_on_linear_fields_and_conserves() -> None:
    from vcell_fenics.core.bulk_remap_mesh import remap_bulk_function_3d

    old, new = _ball_mesh(h=0.3), _ball_mesh(h=0.2)
    V_old = fem.functionspace(old, ("Lagrange", 1))  # type: ignore[arg-type]
    V_new = fem.functionspace(new, ("Lagrange", 1))  # type: ignore[arg-type]
    u = fem.Function(V_old)
    u.interpolate(lambda x: 2.0 + x[0] - 0.5 * x[2])
    moved = remap_bulk_function_3d(u, V_new, conserve=False)
    x = V_new.tabulate_dof_coordinates()
    inside = np.linalg.norm(x, axis=1) < 0.9  # away from where the two boundaries differ
    assert np.allclose(moved.x.array[inside], (2.0 + x[:, 0] - 0.5 * x[:, 2])[inside], atol=1e-10)
    conserved = remap_bulk_function_3d(u, V_new)

    def mass(f: fem.Function) -> float:
        return float(fem.assemble_scalar(fem.form(f * ufl.dx)).real)

    assert mass(conserved) == pytest.approx(mass(u), rel=1e-12)


def test_a_neck_too_thin_to_mesh_is_a_pinch_off_not_a_crash() -> None:
    from vcell_fenics.backend.remesh_3d import remesh_region_3d
    from vcell_fenics.core.region_remesh_netgen import PinchOffError

    # a slab 0.2 thick: remeshed at h = 0.8 the finest allowed size is 0.2, and the region is thinner than
    # twice that — refused before Netgen (which segfaults on such surfaces)
    slab = "geom.x[0]**2 < 0.01 && geom.x[1]**2 < 0.36 && geom.x[2]**2 < 0.36"
    with pytest.raises(PinchOffError, match="neck closing"):
        remesh_region_3d(_ball_mesh(slab, h=0.1, extent=2.0), 0.8)  # type: ignore[arg-type]


def test_snapping_moves_points_onto_the_surface() -> None:
    from vcell_fenics.backend.remesh_3d import snap_to_surface

    # the unit right triangle in z = 0: points above it drop onto it, points off its edges onto the edges
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    query = np.array([[0.2, 0.2, 0.7], [2.0, -1.0, 0.0], [0.8, 0.8, -0.3], [-1.0, -1.0, 1.0]])
    expected = np.array([[0.2, 0.2, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(snap_to_surface(query, points, triangles), expected)
