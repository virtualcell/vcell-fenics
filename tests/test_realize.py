"""Realization v1, step 1: the trivial (dim-0 / well-mixed) case of `backend.realize.realize`.

Checks that a non-spatial GeometryDescription becomes a backend Geometry with one lumped mesh per
compartment (and per membrane), that it satisfies the §1.11.10 cross-check against a matching
MathDescription, and that the unimplemented spatial path and the malformed cases raise clearly.
"""

from __future__ import annotations

import math

import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.backend.realize import RealizationError, realize
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import MathDescription, Subdomain, TemplateEquation, Variable


def _measure(geometry: Geometry, subdomain: str) -> float:
    """The integral of 1 over a subdomain mesh — its area (volume) or, for the membrane, length."""
    mesh = geometry.mesh_of(subdomain)
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh))).real)


def _compartmental(name: str, *subvolumes: str, surfaces: tuple[SurfaceClass, ...] = ()) -> GeometryDescription:
    return GeometryDescription(
        name=name,
        dim=0,
        subvolumes=tuple(SubVolume(name=s, type="compartmental") for s in subvolumes),
        surfaces=surfaces,
    )


def test_trivial_single_compartment() -> None:
    geom = realize(_compartmental("cell", "cytosol"))
    assert geom.name == "cell"
    assert set(geom.subdomains) == {"cytosol"}
    assert geom.kind_of("cytosol") == "volume"
    assert geom.mesh_of("cytosol") is not None


def test_trivial_two_compartments_with_membrane() -> None:
    geom = realize(
        _compartmental(
            "cell",
            "cytosol",
            "extracellular",
            surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
        )
    )
    assert geom.kind_of("cytosol") == "volume"
    assert geom.kind_of("extracellular") == "volume"
    assert geom.kind_of("pm") == "surface"


def test_realized_geometry_passes_cross_check() -> None:
    geom = realize(_compartmental("cell", "cytosol"))
    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cytosol", kind="volume")],
        variables=[Variable(name="u", subdomain="cytosol")],
        equations=[
            TemplateEquation(template="lumped_ode", variable="u", subdomain="cytosol", temporality="time_dependent")
        ],
    )
    assert cross_validate(md, geom) == []


def test_2d_single_subvolume_not_implemented() -> None:
    # A box with no interior partition (one subvolume) isn't the supported interior+background case.
    spatial = GeometryDescription(
        name="cell", dim=2, subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0] < 1"),)
    )
    with pytest.raises(NotImplementedError, match="interior subvolume \\+ background"):
        realize(spatial)


def test_non_compartmental_subvolume_in_dim0_rejected() -> None:
    bad = GeometryDescription(
        name="cell", dim=0, subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0] < 1"),)
    )
    with pytest.raises(RealizationError, match="must be 'compartmental'"):
        realize(bad)


def test_no_subvolumes_rejected() -> None:
    with pytest.raises(RealizationError, match="no subvolumes"):
        realize(GeometryDescription(name="empty", dim=0))


# -- 2D body-fitted, box-partitioned --------------------------------------------


def _disk_in_box(radius: float = 0.5) -> GeometryDescription:
    """A 2x2 box centred at the origin with a disk `cytosol` inside an `extracellular` background,
    meeting at the `pm` membrane."""
    r2 = radius * radius
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {r2}"),
            SubVolume(name="extracellular", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 > {r2}"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
    )


def test_2d_disk_in_box_partition() -> None:
    radius = 0.5
    geom = realize(_disk_in_box(radius), h=0.06, resolution=161)

    assert geom.kind_of("cytosol") == "volume"
    assert geom.kind_of("extracellular") == "volume"
    assert geom.kind_of("pm") == "surface"

    cyto, ext = _measure(geom, "cytosol"), _measure(geom, "extracellular")
    # Body-fitted regions partition the box; the disk area matches pi r^2 within the polygonal
    # discretization of the marched contour (a few percent at this resolution).
    assert cyto + ext == pytest.approx(4.0, abs=1e-9)
    assert cyto == pytest.approx(math.pi * radius**2, rel=0.05)
    assert _measure(geom, "pm") == pytest.approx(2 * math.pi * radius, rel=0.05)


def test_2d_membrane_is_internal_and_faces_external() -> None:
    geom = realize(_disk_in_box(), h=0.08, resolution=121)
    pm = geom.boundary_of("pm")
    assert pm is not None and pm.is_internal and set(pm.subdomains) == {"cytosol", "extracellular"}
    for face in ("x_minus", "x_plus", "y_minus", "y_plus"):
        boundary = geom.boundary_of(face)
        assert boundary is not None and not boundary.is_internal
        assert boundary.subdomains == ("extracellular",)
        assert boundary.facets.size > 0


def test_2d_realized_geometry_passes_cross_check() -> None:
    geom = realize(_disk_in_box(), h=0.08, resolution=121)
    md = MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cytosol", kind="volume"),
            Subdomain(name="extracellular", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[Variable(name="u", subdomain="cytosol")],
        equations=[
            TemplateEquation(template="bulk_radv_diff", variable="u", subdomain="cytosol", temporality="steady_state")
        ],
    )
    assert cross_validate(md, geom) == []


def test_2d_three_subvolumes_not_implemented() -> None:
    three = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="a", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.04"),
            SubVolume(name="b", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.25"),
            SubVolume(name="bg", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 > 0.25"),
        ),
    )
    with pytest.raises(NotImplementedError, match="interior subvolume \\+ background"):
        realize(three)


def test_2d_interior_must_be_analytic() -> None:
    bad = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="a", type="compartmental"),
            SubVolume(name="bg", type="analytic", expression="geom.x[0] > 0"),
        ),
    )
    with pytest.raises(RealizationError, match="must be 'analytic'"):
        realize(bad)


def test_3d_not_implemented() -> None:
    g = GeometryDescription(name="g", dim=3, subvolumes=(SubVolume(name="c", type="compartmental"),))
    with pytest.raises(NotImplementedError, match="v1 supports dim 0 and dim 2"):
        realize(g)
