"""Conforming boundaries of a label field (`backend/label_surfaces.py`): SurfaceNets with box-face
sentinels, orientation, constrained smoothing and the projection onto the smooth interfaces.

The exact invariants are checked exactly — every region's boundary is closed and consistently
oriented, the regions partition the box, box faces are planar, junctions conform — and accuracy
against the analytic shape within the limit a *binary* image sets: the image's own rasterization
error and the smoothing's curvature shrink, both O(pixel)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from numpy.typing import NDArray

from vcell_fenics.backend.label_surfaces import LabelBoundary, extract_boundary
from vcell_fenics.backend.labels import LabelGrid, image_label_grid, smoothed_label_grid
from vcell_fenics.formalism.geometry_schema import GeometryDescription, GeometryImage, PixelClass, SubVolume


def _smoothed(labels_xyz: NDArray[np.uint8], extent: tuple[float, ...]) -> LabelGrid:
    """The smoothed label grid (at the pixel size) of an image given ``[x, y(, z)]``, value k+1 = region k."""

    dim = labels_xyz.ndim
    values = sorted(int(v) for v in np.unique(labels_xyz))
    geometry = GeometryDescription(
        name="g",
        dim=dim,
        extent=(*extent, 1.0) if dim == 2 else extent,  # type: ignore[arg-type]
        subvolumes=tuple(SubVolume(f"r{v}", "image", pixel_value=v) for v in values),
        image=GeometryImage.from_voxels("s", labels_xyz.T, tuple(PixelClass(f"r{v}", v) for v in values)),
    )
    raw = image_label_grid(geometry)
    return smoothed_label_grid(raw, extent=extent, h=min(raw.spacing))


def _lattice(n: int, dim: int) -> tuple[NDArray[np.float64], ...]:
    x = np.linspace(0.0, 1.0, n)
    return tuple(np.meshgrid(*([x] * dim), indexing="ij"))


def _assert_closed_and_oriented(boundary: LabelBoundary) -> None:
    """Every region's boundary, oriented outward, is closed: in 2D each vertex has as many outgoing as
    incoming segments; in 3D each directed edge is matched by its reverse exactly once."""

    for region in {int(p) for p in boundary.pairs.ravel() if p >= 0}:
        mine = (boundary.pairs == region).any(axis=1)
        elements = boundary.elements[mine].copy()
        inward = boundary.pairs[mine][:, 1] == region  # the normal points into pairs[:, 1]
        if boundary.dim == 2:
            # 2D: the left normal points into pairs[:, 1]; walk each region counter-clockwise (left side in)
            elements[~inward] = elements[~inward][:, ::-1]
            out, into = Counter(elements[:, 0].tolist()), Counter(elements[:, 1].tolist())
            assert out == into, f"region {region}: an open boundary"
        else:
            elements[inward] = elements[inward][:, ::-1]
            directed: Counter[tuple[int, int]] = Counter()
            for a, b, c in elements.tolist():
                directed.update([(a, b), (b, c), (c, a)])
            assert all(directed[(v, u)] == count for (u, v), count in directed.items()), f"region {region}"


def test_a_disk_is_closed_partitions_the_box_and_converges() -> None:
    errors = []
    for n in (33, 65, 129):
        xx, yy = _lattice(n, 2)
        grid = _smoothed(np.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.09, 2, 1).astype(np.uint8), (1.0, 1.0))
        boundary = extract_boundary(grid, extent=(1.0, 1.0))
        _assert_closed_and_oriented(boundary)
        assert boundary.region_measure(0) + boundary.region_measure(1) == pytest.approx(1.0, abs=1e-12)
        errors.append(abs(boundary.region_measure(1) - np.pi * 0.09) / (np.pi * 0.09))
    assert max(errors) < 0.01 and errors[-1] < 0.002, errors


def test_a_sphere_converges_within_the_rasterization_limit() -> None:
    volumes, areas = [], []
    for n in (17, 33, 65):
        x, y, z = _lattice(n, 3)
        labels = np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.09, 2, 1).astype(np.uint8)
        boundary = extract_boundary(_smoothed(labels, (1.0, 1.0, 1.0)), extent=(1.0, 1.0, 1.0))
        _assert_closed_and_oriented(boundary)
        assert boundary.region_measure(0) + boundary.region_measure(1) == pytest.approx(1.0, abs=1e-12)
        interface = (boundary.pairs >= 0).all(axis=1)
        volumes.append(abs(boundary.region_measure(1) / (4 / 3 * np.pi * 0.027) - 1))
        areas.append(abs(boundary.measure()[interface].sum() / (4 * np.pi * 0.09) - 1))
    assert volumes == sorted(volumes, reverse=True) and volumes[-1] < 0.025, volumes
    assert areas == sorted(areas, reverse=True) and areas[-1] < 0.015, areas


def test_junctions_and_the_box_conform_in_3d() -> None:
    # two cells touching across a plane (a junction circle ec | a | b) and a sphere cut by x_minus
    n = 41
    x, y, z = (4.0 * c for c in _lattice(n, 3))
    x, y, z = x, y / 2.0, z / 2.0  # a 4 × 2 × 2 box
    labels = np.ones(x.shape, dtype=np.uint8)
    a = (x - 1.5) ** 2 + (y - 1) ** 2 + (z - 1) ** 2 < 0.36
    b = (x - 2.5) ** 2 + (y - 1) ** 2 + (z - 1) ** 2 < 0.36
    labels[a & (x < 2.0)] = 2
    labels[b & (x >= 2.0)] = 3
    labels[x**2 + (y - 1) ** 2 + (z - 1) ** 2 < 0.25] = 4
    boundary = extract_boundary(_smoothed(labels, (4.0, 2.0, 2.0)), extent=(4.0, 2.0, 2.0))
    _assert_closed_and_oriented(boundary)
    total = sum(boundary.region_measure(r) for r in range(4))
    assert total == pytest.approx(16.0, abs=1e-10)
    pairs = {tuple(sorted(p)) for p in boundary.pairs.tolist()}
    assert {(0, 1), (0, 2), (1, 2), (0, 3), (-1, 0), (-1, 3)} <= pairs  # junction pieces and the x_minus cut
    # box faces are exactly planar
    for face, (axis, value) in enumerate([(0, 0.0), (0, 4.0), (1, 0.0), (1, 2.0), (2, 0.0), (2, 2.0)]):
        on_face = np.unique(boundary.elements[(boundary.pairs == -(face + 1)).any(axis=1)])
        assert np.all(boundary.points[on_face, axis] == value)
    # the cut sphere keeps about half a ball of radius 0.5 — only 5 pixels here, so the smoothing's
    # curvature shrink (≈ σ²/R in radius) costs ~10 % of its volume; the box cut itself adds nothing
    assert boundary.region_measure(3) == pytest.approx(2 / 3 * np.pi * 0.125, rel=0.12)


def test_junction_points_conform_in_2d() -> None:
    # three regions meeting at points (a | b | ec), one of them cut by the y_minus edge
    n = 161
    xx, yy = (2.0 * c for c in _lattice(n, 2))
    labels = np.ones(xx.shape, dtype=np.uint8)
    labels[((xx - 0.7) ** 2 + (yy - 1.0) ** 2 < 0.25) & (xx < 1.0)] = 2
    labels[((xx - 1.3) ** 2 + (yy - 1.0) ** 2 < 0.25) & (xx >= 1.0)] = 3
    labels[(xx - 1.0) ** 2 + yy**2 < 0.09] = 4
    boundary = extract_boundary(_smoothed(labels, (2.0, 2.0)), extent=(2.0, 2.0))
    _assert_closed_and_oriented(boundary)
    assert sum(boundary.region_measure(r) for r in range(4)) == pytest.approx(4.0, abs=1e-12)
    # the half disk on y = 0: within the smoothing's curvature shrink at 24 pixels of radius (the same
    # error as the whole disk inside the box — the cut adds none)
    assert boundary.region_measure(3) == pytest.approx(np.pi * 0.09 / 2, rel=0.02)
    on_edge = np.unique(boundary.elements[(boundary.pairs == -3).any(axis=1)])  # y_minus
    assert np.all(boundary.points[on_edge, 1] == 0.0)
