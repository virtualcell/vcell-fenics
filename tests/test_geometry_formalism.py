"""The geometry formalism: schema round-trip and structural validation (no pyvcell)."""

from __future__ import annotations

from vcell_fenics.formalism import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SurfaceClass,
    dump_geometry_yaml,
    load_geometry_yaml,
    validate_geometry,
)


def _cell_in_extracellular() -> GeometryDescription:
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 <= 1.0"),
            SubVolume(name="extracellular", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="membrane", inside="cytosol", outside="extracellular"),),
    )


def _errors(geometry: GeometryDescription) -> list[str]:
    return [d.message for d in validate_geometry(geometry) if d.severity == "error"]


# --- round-trip ----------------------------------------------------------------


def test_yaml_round_trip_is_identity() -> None:
    geometry = _cell_in_extracellular()
    assert load_geometry_yaml(dump_geometry_yaml(geometry)) == geometry


def test_defaults_are_omitted_on_output() -> None:
    geometry = GeometryDescription(name="g", dim=0, subvolumes=(SubVolume(name="cell", type="compartmental"),))
    text = dump_geometry_yaml(geometry)
    assert "origin" not in text  # (0,0,0) default omitted
    assert "extent" not in text  # (1,1,1) default omitted
    assert "surfaces" not in text


def test_image_metadata_round_trips() -> None:
    geometry = GeometryDescription(
        name="img",
        dim=3,
        subvolumes=(SubVolume(name="cell", type="image", pixel_value=1),),
        image=GeometryImage(name="seg", size=(64, 64, 32), pixel_classes=(PixelClass(name="cell", pixel_value=1),)),
    )
    assert load_geometry_yaml(dump_geometry_yaml(geometry)) == geometry


# --- validation ----------------------------------------------------------------


def test_valid_geometry_has_no_errors() -> None:
    assert validate_geometry(_cell_in_extracellular()) == []


def test_bad_dimension_is_rejected() -> None:
    assert any("dimension must be" in m for m in _errors(GeometryDescription(name="g", dim=4)))


def test_surface_referencing_unknown_subvolume_is_rejected() -> None:
    geometry = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(SubVolume(name="a", type="analytic", expression="1.0"),),
        surfaces=(SurfaceClass(name="m", inside="a", outside="nope"),),
    )
    assert any("unknown subvolume 'nope'" in m for m in _errors(geometry))


def test_analytic_subvolume_without_expression_is_rejected() -> None:
    geometry = GeometryDescription(name="g", dim=2, subvolumes=(SubVolume(name="a", type="analytic"),))
    assert any("must have an 'expression'" in m for m in _errors(geometry))


def test_malformed_analytic_expression_is_rejected() -> None:
    geometry = GeometryDescription(
        name="g", dim=2, subvolumes=(SubVolume(name="a", type="analytic", expression="1.0 +"),)
    )
    assert any("malformed analytic expression" in m for m in _errors(geometry))


def test_compartmental_in_spatial_geometry_is_rejected() -> None:
    geometry = GeometryDescription(name="g", dim=2, subvolumes=(SubVolume(name="a", type="compartmental"),))
    assert any("only valid in a non-spatial" in m for m in _errors(geometry))


def test_image_subvolume_without_image_is_rejected() -> None:
    geometry = GeometryDescription(name="g", dim=3, subvolumes=(SubVolume(name="a", type="image", pixel_value=1),))
    assert any("requires the geometry to carry an 'image'" in m for m in _errors(geometry))


def test_duplicate_subvolume_name_is_rejected() -> None:
    geometry = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="a", type="analytic", expression="1.0"),
            SubVolume(name="a", type="analytic", expression="2.0"),
        ),
    )
    assert any("duplicate subvolume name 'a'" in m for m in _errors(geometry))
