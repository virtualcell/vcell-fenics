"""Realizing image geometries (`backend/realize.py`, `_realize_image_partition`): a segmented label
image becomes a body-fitted, tagged mesh — label field (`labels.py`) → conforming boundaries
(`label_surfaces.py`) → Netgen, in 2D and 3D, with any topology: nested regions, regions cut by the box,
and junctions where three subvolumes meet. The realized geometry is the same `Geometry` /
`InterfaceCoupledGeometry` the analytic path produces, so everything downstream is unchanged."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from numpy.typing import NDArray

from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.realize import ImageGeometryWarning, RealizationError, realize, realize_interface_coupled
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


def image_geometry(
    labels_xyz: NDArray[np.uint8],
    extent: tuple[float, float, float],
    names: tuple[str, ...],
    surfaces: tuple[SurfaceClass, ...],
) -> GeometryDescription:
    """An image geometry from a label array indexed ``[x, y(, z)]``: pixel value k+1 is ``names[k]``."""

    return GeometryDescription(
        name="img",
        dim=labels_xyz.ndim,
        extent=extent,
        subvolumes=tuple(SubVolume(name, "image", pixel_value=k + 1) for k, name in enumerate(names)),
        surfaces=surfaces,
        image=GeometryImage.from_voxels("seg", labels_xyz.T, tuple(PixelClass(n, k + 1) for k, n in enumerate(names))),
    )


def two_cells_2d(n: int = 161) -> GeometryDescription:
    """ec with two cells touching along x = 1 (junction points ec | a | b) and a half disk ``c`` cut by
    the y_minus edge, in a 2 × 2 box."""

    x = np.linspace(0.0, 2.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    labels = np.ones(xx.shape, dtype=np.uint8)
    labels[((xx - 0.7) ** 2 + (yy - 1.0) ** 2 < 0.25) & (xx < 1.0)] = 2
    labels[((xx - 1.3) ** 2 + (yy - 1.0) ** 2 < 0.25) & (xx >= 1.0)] = 3
    labels[(xx - 1.0) ** 2 + yy**2 < 0.09] = 4
    surfaces = (
        SurfaceClass("a_ec", "a", "ec"),
        SurfaceClass("b_ec", "b", "ec"),
        SurfaceClass("a_b", "a", "b"),
        SurfaceClass("c_ec", "c", "ec"),
    )
    return image_geometry(labels, (2.0, 2.0, 1.0), ("ec", "a", "b", "c"), surfaces)


def cells_3d(n: int = 41) -> GeometryDescription:
    """ec with a ball ``cell`` inside and a ball ``cut`` cut in half by x_minus, in a 2 × 2 × 2 box."""

    x = np.linspace(0.0, 2.0, n)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    labels = np.ones(xx.shape, dtype=np.uint8)
    labels[(xx - 1.2) ** 2 + (yy - 1.0) ** 2 + (zz - 1.0) ** 2 < 0.25] = 2
    labels[xx**2 + (yy - 1.0) ** 2 + (zz - 1.0) ** 2 < 0.16] = 3
    surfaces = (SurfaceClass("m", "cell", "ec"), SurfaceClass("mc", "cut", "ec"))
    return image_geometry(labels, (2.0, 2.0, 2.0), ("ec", "cell", "cut"), surfaces)


def _measure(geometry: Geometry, subdomain: str) -> float:
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=geometry.mesh_of(subdomain)))).real)


def _min_angle_deg(mesh: dmesh.Mesh) -> float:
    """The smallest interior angle of a triangle mesh (2D) or the smallest dihedral angle (3D)."""

    tdim = mesh.topology.dim
    x = mesh.geometry.x[:, : mesh.geometry.dim]
    cells = mesh.geometry.dofmap
    corners = x[cells]
    if tdim == 2:
        angles = []
        for i in range(3):
            u = corners[:, (i + 1) % 3] - corners[:, i]
            v = corners[:, (i + 2) % 3] - corners[:, i]
            cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
            angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return float(np.min(angles))
    normals = []
    for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
        a, b, c = (corners[:, k] for k in face)
        normal = np.cross(b - a, c - a)
        normals.append(normal / np.linalg.norm(normal, axis=1, keepdims=True))
    dihedral = [
        np.degrees(np.pi - np.arccos(np.clip(np.einsum("ij,ij->i", normals[i], normals[j]), -1.0, 1.0)))
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    return float(np.min(dihedral))


def test_a_2d_image_with_junctions_and_an_edge_cut() -> None:
    geometry = realize(two_cells_2d(), h=0.03)
    areas = {name: _measure(geometry, name) for name in ("ec", "a", "b", "c")}
    assert sum(areas.values()) == pytest.approx(4.0, abs=1e-9)  # the regions tile the box
    assert areas["c"] == pytest.approx(math.pi * 0.09 / 2, rel=0.02)  # the half disk on y = 0
    # every membrane is realized — including a | b, which exists only because of the junctions
    assert _measure(geometry, "a_b") == pytest.approx(0.8, rel=0.05)  # the chord x = 1 of both circles
    assert _measure(geometry, "c_ec") == pytest.approx(math.pi * 0.3, rel=0.02)
    assert geometry.boundaries["a_b"].subdomains == ("a", "b")
    # the y_minus edge is shared by ec and the cut region
    assert geometry.boundaries["y_minus"].subdomains == ("c", "ec")
    assert geometry.parent_mesh is not None and _min_angle_deg(geometry.parent_mesh) > 15.0


def test_a_3d_image_with_an_edge_cut() -> None:
    geometry = realize(cells_3d(), h=0.1)
    volumes = {name: _measure(geometry, name) for name in ("ec", "cell", "cut")}
    assert sum(volumes.values()) == pytest.approx(8.0, abs=1e-9)
    # 5 and 4 pixels of radius: the smoothing's curvature shrink costs ~10 % (it converges with the pixels)
    assert volumes["cell"] == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.12)
    assert volumes["cut"] == pytest.approx(2 / 3 * math.pi * 0.064, rel=0.15)
    assert set(geometry.boundaries["x_minus"].subdomains) == {"cut", "ec"}
    assert _measure(geometry, "m") == pytest.approx(4 * math.pi * 0.25, rel=0.08)
    assert geometry.parent_mesh is not None and _min_angle_deg(geometry.parent_mesh) > 5.0  # no slivers


def test_the_reservoir_wall_is_the_outer_compartments_share_of_the_box() -> None:
    # the inner compartment 'c' touches the box (y_minus): its box facets are not the outer reservoir wall
    coupled = realize_interface_coupled(
        two_cells_2d(),
        inner_subdomain="c",
        outer_subdomain="ec",
        membrane_subdomain="c_ec",
        interface="pm",
        h=0.03,
    )
    ds = ufl.Measure("ds", domain=coupled.parent_mesh, subdomain_data=coupled.facet_tags)
    wall = float(fem.assemble_scalar(fem.form(1.0 * ds(coupled.outer_tag))).real)
    everything = float(fem.assemble_scalar(fem.form(1.0 * ufl.ds(domain=coupled.parent_mesh))).real)
    assert everything == pytest.approx(8.0, abs=1e-9)  # the whole box perimeter
    # the wall is the perimeter minus c's footprint on y = 0 (its diameter 0.6, a little less smoothed)
    assert 8.0 - 0.6 < wall < 8.0 - 0.58


def test_a_touching_pair_without_a_surface_class_is_reported() -> None:
    description = two_cells_2d()
    missing = GeometryDescription(
        name=description.name,
        dim=2,
        extent=description.extent,
        subvolumes=description.subvolumes,
        surfaces=tuple(s for s in description.surfaces if s.name != "a_b"),
        image=description.image,
    )
    with pytest.warns(ImageGeometryWarning, match="'a' and 'b' touch but no surface class"):
        geometry = realize(missing, h=0.03)
    assert "a_b" not in geometry.boundaries


def test_an_image_without_voxels_cannot_be_meshed() -> None:
    description = two_cells_2d()
    assert description.image is not None
    metadata_only = GeometryDescription(
        name="img",
        dim=2,
        extent=description.extent,
        subvolumes=description.subvolumes,
        surfaces=description.surfaces,
        image=GeometryImage(name="seg", size=description.image.size, pixel_classes=description.image.pixel_classes),
    )
    with pytest.raises(RealizationError, match="no image voxels"):
        realize(metadata_only, h=0.03)


def test_a_3d_mesh_too_fine_to_build_is_refused() -> None:
    with pytest.raises(RealizationError, match="would need"):
        realize(cells_3d(), h=0.005)


def test_the_vcell_image_fixture() -> None:
    # VCell's image3d geometry (256×256×34, ec ⊃ cytosol ⊃ Nucleus) at h = 2 µm: every region and membrane
    # is realized, the regions tile the 74.24 × 74.24 × 26 µm box, and the volumes track VCell's own
    # (124712.10 / 14891.9 / 3697.0 µm³) — within ~8 % at this coarse h (the h = 1 µm check is in the
    # integration suite)
    geometry = import_geometry(read_simtask(_FIXTURE).geometry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImageGeometryWarning)  # at h = 2 the nucleus reaches ec
        realized = realize(geometry, h=2.0)
    volumes = {name: _measure(realized, name) for name in ("ec", "cytosol", "Nucleus")}
    assert sum(volumes.values()) == pytest.approx(74.24 * 74.24 * 26.0, rel=1e-12)
    for name, vcell in (("ec", 124712.10), ("cytosol", 14891.9), ("Nucleus", 3697.0)):
        assert volumes[name] == pytest.approx(vcell, rel=0.09), name
    assert {"cytosol_ec_membrane", "Nucleus_cytosol_membrane"} <= set(realized.boundaries)


@pytest.mark.integration
def test_the_vcell_image_fixture_at_1um() -> None:
    # at h = 1 µm (~740k tetrahedra, ~1 min): volumes within 3 % of VCell's, and membrane areas within 6 %
    # of VCell's stored 4738.64 / 1406.77 µm² (VCell's are areas of its smoothed voxel surfaces)
    realized = realize(import_geometry(read_simtask(_FIXTURE).geometry), h=1.0)
    for name, vcell in (("ec", 124712.10), ("cytosol", 14891.9), ("Nucleus", 3697.0)):
        assert _measure(realized, name) == pytest.approx(vcell, rel=0.03), name
    for name, vcell in (("cytosol_ec_membrane", 4738.64), ("Nucleus_cytosol_membrane", 1406.77)):
        assert _measure(realized, name) == pytest.approx(vcell, rel=0.06), name
