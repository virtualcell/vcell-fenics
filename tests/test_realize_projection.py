"""The marching path's membrane lies on the analytic shape (#249).

`_realize_2d_partition` / `_realize_3d_partition` march each shape's ``φ = 0`` on a grid, simplify it and
hand it to Netgen, which resamples it; every step leaves the boundary off the curve by an amount that
fell only about as ``h``. After meshing, each membrane vertex is projected onto its shape's exact
``φ = 0`` (`realize._project_interfaces`), at time zero only.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend.implicit_fields import eval_field
from vcell_fenics.backend.realize import GeometryProjectionWarning, _project_interfaces, realize
from vcell_fenics.formalism.expr import Expr
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import lower_predicate

_DISK = "geom.x[0]**2 + geom.x[1]**2 < 1.0"
_ELLIPSE = "(geom.x[0] / 1.2)**2 + (geom.x[1] / 0.7)**2 < 1.0"
_RING = "geom.x[0]**2 + geom.x[1]**2 < 1.0 && geom.x[0]**2 + geom.x[1]**2 > 0.3"
_SPHERE = "geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < 1.0"


def _two_regions(expression: str, dim: int, half: float = 2.0) -> GeometryDescription:
    return GeometryDescription(
        name="shape",
        dim=dim,
        extent=(2 * half, 2 * half, 2 * half if dim == 3 else 1.0),
        origin=(-half, -half, -half if dim == 3 else 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression=expression),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )


def _distance_off(expression: str, points: np.ndarray) -> np.ndarray:
    """|φ| / |∇φ| at each point: a first-order distance to the shape's zero set."""

    field = lower_predicate(parse(expression))
    dim = points.shape[1]
    phi = eval_field(field, tuple(points[:, k] for k in range(dim)))
    grad = np.zeros_like(points)
    eps = 1e-7
    for k in range(dim):
        plus, minus = points.copy(), points.copy()
        plus[:, k] += eps
        minus[:, k] -= eps
        grad[:, k] = (
            eval_field(field, tuple(plus[:, j] for j in range(dim)))
            - eval_field(field, tuple(minus[:, j] for j in range(dim)))
        ) / (2 * eps)
    result: np.ndarray = np.abs(phi) / np.linalg.norm(grad, axis=1)
    return result


@pytest.mark.parametrize("expression", [_DISK, _ELLIPSE, _RING], ids=["disk", "ellipse", "ring"])
@pytest.mark.parametrize("h", [0.2, 0.1])
def test_every_2d_membrane_vertex_lies_on_the_shape(expression: str, h: float) -> None:
    geometry = realize(_two_regions(expression, 2), h=h)
    points = geometry.mesh_of("pm").geometry.x[:, :2]
    assert points.shape[0] > 20
    assert _distance_off(expression, points).max() < 1e-9

    # the cells stay the right way round: the region areas still tile the box
    total = sum(
        float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=geometry.mesh_of(name)))).real)
        for name in ("cell", "ec")
    )
    assert total == pytest.approx(16.0, rel=1e-12)


def test_the_3d_membrane_lies_on_the_sphere_and_its_volume_converges_at_second_order() -> None:
    errors = []
    for h in (0.4, 0.2):
        geometry = realize(_two_regions(_SPHERE, 3, half=1.5), h=h)
        points = geometry.mesh_of("pm").geometry.x
        assert np.abs(np.linalg.norm(points, axis=1) - 1.0).max() < 1e-12
        volume = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=geometry.mesh_of("cell")))).real)
        errors.append(abs(volume - 4.0 * math.pi / 3.0))
    # facets inscribed in the sphere: an O(h²) deficit (before #249 the ratio was about 3.1)
    assert errors[0] / errors[1] > 3.6


def _flat(expression: str) -> list[Expr]:
    return [lower_predicate(parse(expression))]


def test_a_move_that_would_invert_a_cell_is_pulled_back() -> None:
    # v0 is the one membrane vertex (it touches material 2 above and material 1 below). Its shape is the
    # line y = 0.3; moving it there would push it through the edge v1–v2 at y = 0.1 and invert cell 0.
    points = np.array([[0.0, 0.0], [1.0, 0.1], [-1.0, 0.1], [-1.0, -1.0], [1.0, -1.0]])
    cells = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    material = np.array([2, 1], dtype=np.int32)
    out = _project_interfaces(points, cells, material, _flat("geom.x[1] < 0.3"), [1], h=2.0)

    assert out[0, 0] == pytest.approx(0.0)
    assert 0.0 < out[0, 1] < 0.1 * (1.0 - 0.2)  # moved toward the line, but the cell keeps ≥ 20 % of its area
    np.testing.assert_array_equal(out[1:], points[1:])


def test_a_vertex_with_no_zero_nearby_stays_where_netgen_put_it() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.1], [-1.0, 0.1], [-1.0, -1.0], [1.0, -1.0]])
    cells = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    material = np.array([2, 1], dtype=np.int32)

    # φ ≥ 1 everywhere: Newton cannot converge, and says so
    with pytest.warns(GeometryProjectionWarning, match="1 membrane vertices"):
        out = _project_interfaces(points, cells, material, _flat("geom.x[0]**2 + geom.x[1]**2 + 1.0 < 0.0"), [1], h=2.0)
    np.testing.assert_array_equal(out, points)

    # a zero, but farther than a quarter of h: detail the marching rounded off, left without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeometryProjectionWarning)
        out = _project_interfaces(points, cells, material, _flat("geom.x[1] < 5.0"), [1], h=2.0)
    np.testing.assert_array_equal(out, points)


def _teeth(teeth: int) -> str:
    d2 = "(geom.x[0])**2 + (geom.x[1])**2"
    return f"{d2} < 6.25 + (1 + 0.4*sin({teeth}*atan2(geom.x[1], geom.x[0])))**2"


def test_detail_finer_than_the_mesh_is_left_rounded_and_resolved_detail_is_projected() -> None:
    """35 teeth, crests of radius 0.034: at h = 0.1 the crests are finer than the mesh and keep the marched
    rounding (projecting some of their vertices thinned the cells beside them), while the flanks are projected.
    At h = 0.025 the crests are resolved and every vertex is projected."""

    expression = _teeth(35)
    coarse = realize(_two_regions(expression, 2, half=3.5), h=0.1)
    off = _distance_off(expression, coarse.mesh_of("pm").geometry.x[:, :2])
    assert 0.05 * off.size < (off > 1e-9).sum() < 0.5 * off.size
    parent = coarse.parent_mesh
    assert parent is not None
    xp, cells = parent.geometry.x[:, :2], parent.geometry.dofmap
    u, v = xp[cells[:, 1]] - xp[cells[:, 0]], xp[cells[:, 2]] - xp[cells[:, 0]]
    assert (0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])).min() > 1.5e-4  # 2.2e-4 measured; 1.6e-4 if forced

    fine = realize(_two_regions(expression, 2, half=3.5), h=0.025)
    assert _distance_off(expression, fine.mesh_of("pm").geometry.x[:, :2]).max() < 1e-9


def test_a_shape_far_from_the_origin_is_projected() -> None:
    """The tolerance is relative to h: an expanded polynomial at large coordinates rounds φ to ~1e-8."""

    expanded = "geom.x[0]*geom.x[0] - 6000*geom.x[0] + geom.x[1]*geom.x[1] - 6000*geom.x[1] + 17999900 < 0"
    description = GeometryDescription(
        name="far",
        dim=2,
        extent=(40.0, 40.0, 1.0),
        origin=(2980.0, 2980.0, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression=expanded),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeometryProjectionWarning)
        geometry = realize(description, h=1.0)
    r = np.linalg.norm(geometry.mesh_of("pm").geometry.x[:, :2] - 3000.0, axis=1)
    assert np.abs(r - 10.0).max() < 1e-7
