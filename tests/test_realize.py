"""Realization v1, step 1: the trivial (dim-0 / well-mixed) case of `backend.realize.realize`.

Checks that a non-spatial GeometryDescription becomes a backend Geometry with one lumped mesh per
compartment (and per membrane), that it satisfies the §1.11.10 cross-check against a matching
MathDescription, and that the unimplemented spatial path and the malformed cases raise clearly.
"""

from __future__ import annotations

import pytest

from vcell_fenics.backend.geometry import cross_validate
from vcell_fenics.backend.realize import RealizationError, realize
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import MathDescription, Subdomain, TemplateEquation, Variable


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


def test_spatial_geometry_not_implemented() -> None:
    spatial = GeometryDescription(
        name="cell", dim=2, subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0] < 1"),)
    )
    with pytest.raises(NotImplementedError, match="body-fitted"):
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
