"""The geometry formalism: schema round-trip and structural validation (no pyvcell)."""

from __future__ import annotations

import numpy as np
import pytest

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


def _two_cells() -> GeometryDescription:
    # a 2D segmented image: ec = 1 with two cells (2 and 3) — the layout VCell writes, one byte per pixel
    labels = np.ones((6, 8), dtype=np.uint8)
    labels[1:4, 1:4] = 2
    labels[2:5, 4:7] = 3
    return GeometryDescription(
        name="img",
        dim=2,
        extent=(7.0, 5.0, 1.0),
        subvolumes=(
            SubVolume(name="ec", type="image", pixel_value=1),
            SubVolume(name="a", type="image", pixel_value=2),
            SubVolume(name="b", type="image", pixel_value=3),
        ),
        image=GeometryImage.from_voxels("seg", labels, (PixelClass("ec", 1), PixelClass("a", 2), PixelClass("b", 3))),
    )


def test_image_voxels_round_trip() -> None:
    geometry = _two_cells()
    assert geometry.image is not None and geometry.image.size == (8, 6, 1)
    voxels = geometry.image.voxels()
    assert voxels.shape == (1, 6, 8) and voxels[0, 2, 5] == 3 and voxels[0, 0, 0] == 1
    assert load_geometry_yaml(dump_geometry_yaml(geometry)) == geometry
    assert validate_geometry(geometry) == []


def test_image_without_voxels_cannot_decode() -> None:
    with pytest.raises(ValueError, match="carries no voxel data"):
        GeometryImage(name="seg", size=(2, 2, 1)).voxels()


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


def test_image_voxels_that_do_not_fit_the_size_are_rejected() -> None:
    geometry = _two_cells()
    assert geometry.image is not None
    wrong = GeometryImage(name="seg", size=(8, 7, 1), compressed_content=geometry.image.compressed_content)
    bad = GeometryDescription(name="g", dim=2, subvolumes=geometry.subvolumes, image=wrong)
    assert any("size (8, 7, 1) needs 56" in m for m in _errors(bad))


def test_unmapped_pixel_value_is_rejected_and_empty_subvolume_warned() -> None:
    geometry = _two_cells()
    # drop 'b' (pixel 3 now owned by nobody) and add an image subvolume whose value never occurs
    subvolumes = (*geometry.subvolumes[:2], SubVolume(name="c", type="image", pixel_value=9))
    findings = validate_geometry(
        GeometryDescription(name="g", dim=2, extent=geometry.extent, subvolumes=subvolumes, image=geometry.image)
    )
    assert any(d.severity == "error" and "pixel value 3 occurs" in d.message for d in findings)
    assert any(d.severity == "warning" and "pixel_value 9 occurs in no voxel" in d.message for d in findings)
