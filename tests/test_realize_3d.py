"""Verification of 3D body-fitted realization (`backend/realize`, dim 3).

`realize()` partitions a 3D bounding box by marching-cubes isosurfaces of the subvolumes'
implicit fields and volume-meshes it with Netgen (ADR 008 §8: CSG `OrthoBrick` box +
`STLGeometry`-remeshed surfaces + `FaceDescriptor` merge). The checks mirror the 2D suite:

1. **Single subvolume** is the whole bounding box — a tetrahedral box mesh, one volume
   region, the six box faces named.
2. **Sphere in box** partitions into two conforming volume regions plus a named membrane;
   the regions tile the box exactly and the inner region approximates the sphere volume.
"""

from __future__ import annotations

import math

import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass


def _measure(geometry: Geometry, subdomain: str) -> float:
    """The integral of 1 over a subdomain mesh — its volume, or the membrane's area."""
    mesh = geometry.mesh_of(subdomain)
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh))).real)


def _sphere_in_box(radius: float = 1.0) -> GeometryDescription:
    """A 4×4×4 box centred at the origin with a spherical `cytosol` inside an `extracellular`
    background, joined by a `pm` membrane."""
    r2 = radius * radius
    return GeometryDescription(
        name="cell3d",
        dim=3,
        extent=(4.0, 4.0, 4.0),
        origin=(-2.0, -2.0, -2.0),
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < {r2}"),
            SubVolume(
                name="extracellular", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 > {r2}"
            ),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
    )


# ---------------------------------------------------------------------------
# 1. single subvolume ⇒ whole box, six faces named
# ---------------------------------------------------------------------------


def test_3d_single_subvolume_is_the_whole_box() -> None:
    spatial = GeometryDescription(
        name="cube",
        dim=3,
        extent=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        subvolumes=(SubVolume(name="domain", type="analytic", expression="1.0"),),
    )
    geom = realize(spatial, h=0.5)

    assert set(geom.subdomains) == {"domain"}
    assert geom.kind_of("domain") == "volume"
    assert _measure(geom, "domain") == pytest.approx(8.0, rel=1e-6)

    assert set(geom.boundaries) == {"x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus"}
    for face in ("x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus"):
        boundary = geom.boundary_of(face)
        assert boundary is not None and not boundary.is_internal and boundary.subdomains == ("domain",)
        assert boundary.facets.size > 0


# ---------------------------------------------------------------------------
# 2. sphere in box ⇒ two conforming regions + named membrane
# ---------------------------------------------------------------------------


def test_3d_sphere_in_box_partition() -> None:
    radius = 1.0
    geom = realize(_sphere_in_box(radius), h=0.35)

    assert geom.kind_of("cytosol") == "volume"
    assert geom.kind_of("extracellular") == "volume"
    assert geom.kind_of("pm") == "surface"

    cyto, ext = _measure(geom, "cytosol"), _measure(geom, "extracellular")
    assert cyto + ext == pytest.approx(64.0, rel=1e-6)  # the regions tile the box exactly
    assert cyto == pytest.approx(4.0 / 3.0 * math.pi * radius**3, rel=0.15)  # faceted sphere, coarse mesh

    pm = geom.boundary_of("pm")
    assert pm is not None and pm.is_internal and set(pm.subdomains) == {"cytosol", "extracellular"}
    assert pm.facets.size > 0
