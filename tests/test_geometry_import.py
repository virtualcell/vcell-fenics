"""The VCell geometry importer: `pyvcell.vcml.models_geometry.Geometry` → `GeometryDescription`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as g

from vcell_fenics.formalism import dump_geometry_yaml, load_geometry_yaml, validate_geometry
from vcell_fenics.pyvcell_bridge import import_geometry
from vcell_fenics.pyvcell_bridge.simtask import read_simtask

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "simtask"


def test_analytic_cell_in_extracellular_imports() -> None:
    geom = g.Geometry(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        subvolumes=[
            g.SubVolume(
                name="cytosol", handle=0, subvolume_type=g.SubVolumeType.analytic, analytic_expr="x^2 + y^2 < 1"
            ),
            g.SubVolume(name="ext", handle=1, subvolume_type=g.SubVolumeType.analytic, analytic_expr="1.0"),
        ],
        surface_classes=[g.SurfaceClass(name="membrane", subvolume_ref_1="cytosol", subvolume_ref_2="ext")],
    )
    gd = import_geometry(geom)

    assert gd.name == "cell" and gd.dim == 2 and gd.extent == (2.0, 2.0, 1.0)
    assert [(s.name, s.type) for s in gd.subvolumes] == [("cytosol", "analytic"), ("ext", "analytic")]
    # the analytic expression is translated through the math coordinate rules (x/y → geom.x[0..1], ^ → **)
    assert gd.subvolumes[0].expression == "geom.x[0]**2 + geom.x[1]**2 < 1"
    assert [(s.name, s.inside, s.outside) for s in gd.surfaces] == [("membrane", "cytosol", "ext")]
    assert [d for d in validate_geometry(gd) if d.severity == "error"] == []


def test_compartmental_nonspatial_imports() -> None:
    geom = g.Geometry(
        name="wellmixed",
        dim=0,
        subvolumes=[g.SubVolume(name="cell", handle=0, subvolume_type=g.SubVolumeType.compartmental)],
    )
    gd = import_geometry(geom)
    assert gd.dim == 0 and [(s.name, s.type) for s in gd.subvolumes] == [("cell", "compartmental")]
    assert [d for d in validate_geometry(gd) if d.severity == "error"] == []


def test_image_geometry_without_voxels_imports_its_metadata() -> None:
    geom = g.Geometry(
        name="img",
        dim=3,
        subvolumes=[g.SubVolume(name="cell", handle=0, subvolume_type=g.SubVolumeType.image, image_pixel_value=1)],
        image=g.Image(
            name="seg",
            size=(64, 64, 32),
            uncompressed_size=0,
            compressed_content="",  # a metadata-only source (e.g. the parsed corpus strips the blob)
            pixel_classes=[g.PixelClass(name="cell", pixel_value=1)],
        ),
    )
    gd = import_geometry(geom)
    assert gd.subvolumes[0].type == "image" and gd.subvolumes[0].pixel_value == 1
    assert gd.image is not None and gd.image.name == "seg" and gd.image.size == (64, 64, 32)
    assert [(p.name, p.pixel_value) for p in gd.image.pixel_classes] == [("cell", 1)]
    assert gd.image.compressed_content is None
    assert [d for d in validate_geometry(gd) if d.severity == "error"] == []


def test_a_simtask_image_geometry_imports_its_voxels() -> None:
    # VCell's image3d fixture: ec / cytosol / Nucleus as pixel values 1 / 2 / 3 in a 256×256×34 image
    task = read_simtask(_FIXTURES / "image3d_SimID_274630052_0__0.simtask.xml")
    gd = import_geometry(task.geometry)
    assert [(s.name, s.type, s.pixel_value) for s in gd.subvolumes] == [
        ("ec", "image", 1),
        ("cytosol", "image", 2),
        ("Nucleus", "image", 3),
    ]
    assert gd.image is not None and gd.image.size == (256, 256, 34)
    voxels = gd.image.voxels()
    np.testing.assert_array_equal(voxels, task.geometry.image.ndarray_3d_u8)  # pyvcell's own decode
    values, counts = np.unique(voxels, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {1: 1949790, 2: 223074, 3: 55360}
    assert load_geometry_yaml(dump_geometry_yaml(gd)) == gd
    assert [d for d in validate_geometry(gd) if d.severity == "error"] == []


def test_name_override() -> None:
    geom = g.Geometry(name="vcell_name", dim=0)
    assert import_geometry(geom, name="disk_2d").name == "disk_2d"
