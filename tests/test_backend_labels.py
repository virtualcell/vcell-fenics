"""The label field of an image geometry (`backend/labels.py`): VCell's vertex-centred lattice, the
analytic overlay, smoothing + resampling, speck and pinch clean-up, and the topology report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from vcell_fenics.backend.implicit_fields import RealizationError
from vcell_fenics.backend.labels import (
    adjacent_pairs,
    component_counts,
    drop_fragments,
    find_pinches,
    image_label_grid,
    label_geometry,
    repair_pinches,
    smoothed_label_grid,
)
from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SurfaceClass,
)
from vcell_fenics.pyvcell_bridge import import_geometry
from vcell_fenics.pyvcell_bridge.simtask import read_simtask

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simtask" / "image3d_SimID_274630052_0__0.simtask.xml"


def _image_geometry(
    pixels_yx: NDArray[np.uint8],
    extent: tuple[float, float, float],
    names: tuple[str, ...],
    surfaces: tuple[SurfaceClass, ...] = (),
    extra: tuple[SubVolume, ...] = (),
) -> GeometryDescription:
    """A geometry over a label image given as VCell stores it: ``(ny, nx)`` (2D) or ``(nz, ny, nx)``;
    pixel value ``k + 1`` is subvolume ``names[k]``."""

    dim = 2 if pixels_yx.ndim == 2 else 3
    classes = tuple(PixelClass(name, k + 1) for k, name in enumerate(names))
    subvolumes = (*extra, *(SubVolume(name=name, type="image", pixel_value=k + 1) for k, name in enumerate(names)))
    return GeometryDescription(
        name="img",
        dim=dim,
        extent=extent,
        subvolumes=subvolumes,
        surfaces=surfaces,
        image=GeometryImage.from_voxels("seg", pixels_yx, classes),
    )


def _disk_image(n: int, radius: float) -> NDArray[np.uint8]:
    """A disk (pixel value 2) of ``radius`` centred in the unit box, on VCell's vertex-centred lattice."""

    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")  # (ny, nx), as VCell stores it
    return np.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < radius**2, 2, 1).astype(np.uint8)


def test_the_lattice_is_vertex_centred_and_indexed_x_first() -> None:
    pixels = np.ones((4, 6), dtype=np.uint8)  # ny = 4, nx = 6
    pixels[1, 5] = 2  # y = 1, x = 5
    grid = image_label_grid(_image_geometry(pixels, (5.0, 3.0, 1.0), ("ec", "cell")))
    assert grid.labels.shape == (6, 4)
    assert grid.labels[5, 1] == 1 and int(grid.labels.sum()) == 1  # 'cell' is subvolume index 1
    assert grid.spacing == (1.0, 1.0)  # extent / (n − 1): the first and last pixels are on the faces
    assert grid.axis(0)[-1] == 5.0 and grid.axis(1)[-1] == 3.0


def test_analytic_subvolumes_overwrite_the_image_earliest_first() -> None:
    pixels = np.ones((11, 11), dtype=np.uint8)
    wide = SubVolume(name="wide", type="analytic", expression="geom.x[0] < 0.55")
    narrow = SubVolume(name="narrow", type="analytic", expression="geom.x[0] < 0.25")
    grid = image_label_grid(_image_geometry(pixels, (1.0, 1.0, 1.0), ("ec",), extra=(narrow, wide)))
    column = grid.labels[:, 5]  # nodes at x = 0.0, 0.1, …, 1.0
    assert column.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # narrow wins over wide; the image elsewhere


def test_unmapped_pixel_values_are_refused() -> None:
    pixels = np.ones((3, 3), dtype=np.uint8)
    pixels[1, 1] = 7
    with pytest.raises(RealizationError, match=r"pixel values \[7\]"):
        image_label_grid(_image_geometry(pixels, (1.0, 1.0, 1.0), ("ec",)))


def test_smoothing_converges_to_the_disk() -> None:
    # the smoothed, resampled disk's area (node count × node area) converges to πr² as the image refines
    radius, errors = 0.3, []
    for n in (33, 65, 129):
        raw = image_label_grid(_image_geometry(_disk_image(n, radius), (1.0, 1.0, 1.0), ("ec", "cell")))
        h = 1.0 / (n - 1) / 2  # resample finer than the pixels: the smoothing, not the grid, sets the shape
        grid = smoothed_label_grid(raw, extent=(1.0, 1.0), h=h)
        area = float((grid.labels == 1).sum()) * grid.spacing[0] * grid.spacing[1]
        errors.append(abs(area - np.pi * radius**2) / (np.pi * radius**2))
    assert errors[-1] < 0.01 and errors[0] > errors[-1], errors


def test_checkerboard_and_corner_pinches_are_repaired() -> None:
    square = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)  # two 0s touching at a corner
    assert find_pinches(square).any()
    repaired = repair_pinches(square)
    assert not find_pinches(repaired).any()

    cube = np.ones((3, 3, 3), dtype=np.int32)
    cube[0, 0, 0] = cube[1, 1, 1] = 0  # a vertex-only contact across a cube diagonal
    assert find_pinches(cube).any()
    assert not find_pinches(repair_pinches(cube)).any()


def test_specks_are_absorbed_but_a_severed_half_is_kept() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 1  # a second large piece of subvolume 1
    labels[10, 2] = 1  # a one-node speck
    cleaned = drop_fragments(labels, {0: 1, 1: 1})  # the image had one piece of each
    assert component_counts(cleaned, 2) == [1, 2]  # the speck is gone; the large severed piece stays
    assert cleaned[10, 2] == 0


def test_the_topology_report() -> None:
    # a one-pixel-wide sliver vanishes at a coarse mesh size — an error, not a silent loss
    pixels = np.ones((41, 41), dtype=np.uint8)
    pixels[:, 20] = 2
    sliver = _image_geometry(pixels, (1.0, 1.0, 1.0), ("ec", "sliver"))
    with pytest.raises(RealizationError, match=r"\['sliver'\] vanish"):
        label_geometry(sliver, h=0.1)

    # two touching cells with a declared membrane each against ec, but none between them
    pixels = np.ones((41, 41), dtype=np.uint8)
    pixels[10:30, 5:20] = 2
    pixels[10:30, 20:35] = 3
    surfaces = (SurfaceClass("a_ec", "a", "ec"), SurfaceClass("b_ec", "b", "ec"))
    report = label_geometry(_image_geometry(pixels, (1.0, 1.0, 1.0), ("ec", "a", "b"), surfaces), h=0.025)
    assert report.components == (1, 1, 1)
    assert adjacent_pairs(report.grid.labels) == {(0, 1), (0, 2), (1, 2)}
    assert report.warnings == ("subvolumes 'a' and 'b' touch but no surface class names that membrane",)


def test_the_vcell_image_fixture() -> None:
    # 256×256×34, anisotropic voxels (0.29 × 0.29 × 0.79 µm): nested ec ⊃ cytosol ⊃ Nucleus
    geometry = import_geometry(read_simtask(_FIXTURE).geometry)
    report = label_geometry(geometry, h=1.0)
    assert report.grid.labels.shape == (75, 75, 27)
    assert report.components == (1, 1, 1) and report.warnings == ()
    assert not find_pinches(report.grid.labels).any()
    assert adjacent_pairs(report.grid.labels) == {(0, 1), (1, 2)}  # ec|cytosol and cytosol|Nucleus only
    # at h = 2 µm the cytosol between the nucleus and ec is thinner than the grid: they touch, and it says so
    coarse = label_geometry(geometry, h=2.0)
    assert coarse.warnings == ("subvolumes 'ec' and 'Nucleus' touch but no surface class names that membrane",)
