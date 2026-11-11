"""Remeshing a moving partition (ADR 016 decisions 1, 3, 4): the whole partition meshed afresh from its own
skeleton keeps every region exactly, so every compartment's area, every membrane's length and every species'
total are kept to round-off, a linear field is reproduced in every compartment, and a membrane's values are
exact (kept nodes keep theirs, added nodes lie on the old segments).

Cases: nested loops, a membrane ending on the box, three regions meeting at a junction, and a partition moved
and stretched so that Netgen adds points on its stretched segments.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh

from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.backend.partition_remesh import (
    RemeshLostMass,
    _check_total,
    remesh_partition,
    skeleton,
    transfer_compartment,
    transfer_membrane,
)
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_R2 = "geom.x[0]**2 + geom.x[1]**2"


def _box(name: str, subvolumes: tuple[SubVolume, ...], surfaces: tuple[SurfaceClass, ...]) -> GeometryDescription:
    return GeometryDescription(
        name=name, dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0), subvolumes=subvolumes, surfaces=surfaces
    )


def _nested() -> GeometryDescription:
    return _box(
        "nested",
        (
            SubVolume(name="nuc", type="analytic", expression=f"{_R2} < 0.09"),
            SubVolume(name="cyto", type="analytic", expression=f"{_R2} < 0.4"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        (SurfaceClass(name="ne", inside="nuc", outside="cyto"), SurfaceClass(name="pm", inside="cyto", outside="ext")),
    )


def _touching_box() -> GeometryDescription:
    return _box(
        "strip",
        (
            SubVolume(name="left", type="analytic", expression="geom.x[0] < -0.2 + 0.2*geom.x[1]"),
            SubVolume(name="right", type="analytic", expression="1.0"),
        ),
        (SurfaceClass(name="wall", inside="left", outside="right"),),
    )


def _junction() -> GeometryDescription:
    return _box(
        "junction",
        (
            SubVolume(name="a", type="analytic", expression="geom.x[0] < 0.1"),
            SubVolume(name="b", type="analytic", expression="geom.x[1] > 0.2"),
            SubVolume(name="c", type="analytic", expression="1.0"),
        ),
        (
            SurfaceClass(name="ab", inside="a", outside="b"),
            SurfaceClass(name="ac", inside="a", outside="c"),
            SurfaceClass(name="bc", inside="b", outside="c"),
        ),
    )


def _measure(mesh: dmesh.Mesh) -> float:
    return float(np.real(fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))))


def _linear(V: fem.FunctionSpace) -> fem.Function:
    u = fem.Function(V)
    u.interpolate(lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1])
    return u


def _check_remesh(old: MultiCompartmentGeometry, new: MultiCompartmentGeometry) -> None:
    """Every region kept, every field carried: the shared assertions of each case."""
    assert set(new.compartments) == set(old.compartments)
    assert set(new.membranes) == set(old.membranes)
    for name, part in old.compartments.items():
        assert _measure(new.compartments[name].mesh) == pytest.approx(_measure(part.mesh), rel=1e-12)
        V_old = fem.functionspace(part.mesh, ("Lagrange", 1))
        V_new = fem.functionspace(new.compartments[name].mesh, ("Lagrange", 1))
        rough = fem.Function(V_old)
        rough.x.array[:] = np.random.default_rng(1).uniform(0.5, 1.5, rough.x.array.size)
        moved, kinds = transfer_compartment({"lin": _linear(V_old), "rough": rough}, V_new)
        assert kinds["lin"] == "projection"
        assert np.abs(moved["lin"].x.array - _linear(V_new).x.array).max() < 1e-10  # linear fields are exact
    for name, membrane in old.membranes.items():
        assert _measure(new.membranes[name].mesh) == pytest.approx(_measure(membrane.mesh), rel=1e-12)
        V_old = fem.functionspace(membrane.mesh, ("Lagrange", 1))
        V_new = fem.functionspace(new.membranes[name].mesh, ("Lagrange", 1))
        moved_m = transfer_membrane({"lin": _linear(V_old)}, V_new)
        # an affine function is affine along each straight old segment, so the transfer reproduces it at every node
        assert np.abs(moved_m["lin"].x.array - _linear(V_new).x.array).max() < 1e-12
    # conforming: every exterior facet of the new parent lies on the box (a crack or hanging node would not)
    parent = new.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    exterior = dmesh.exterior_facet_indices(parent.topology)
    midpoints = dmesh.compute_midpoints(parent, tdim - 1, exterior)[:, :2]
    on_box = np.isclose(np.abs(midpoints), 1.0, atol=1e-12).any(axis=1)
    assert on_box.all()


@pytest.mark.parametrize("description", [_nested(), _touching_box(), _junction()], ids=lambda d: d.name)
def test_a_remesh_of_an_unmoved_partition_keeps_every_region_and_field(description: GeometryDescription) -> None:
    old = realize_multi_compartment(description, h=0.1)
    new = remesh_partition(old, description, h=0.1)
    _check_remesh(old, new)


def test_the_skeleton_puts_each_segments_left_region_on_its_left() -> None:
    old = realize_multi_compartment(_nested(), h=0.15)
    boundary = skeleton(old)
    p, q = boundary.points[boundary.elements[:, 0]], boundary.points[boundary.elements[:, 1]]
    # a point just left of each segment's midpoint lies in a cell of the region named on its left (the mesh's own
    # cells: its membranes are polygons, whose chords lie inside the analytic circles)
    from dolfinx import geometry as dgeom

    d = q - p
    probe = 0.5 * (p + q) + 1e-4 * np.stack([-d[:, 1], d[:, 0]], axis=1) / np.linalg.norm(d, axis=1)[:, None]
    parent = old.parent_mesh
    points = np.column_stack([probe, np.zeros(len(probe))])
    tree = dgeom.bb_tree(parent, parent.topology.dim)
    hits = dgeom.compute_colliding_cells(parent, dgeom.compute_collisions_points(tree, points), points)
    region_of_cell = np.zeros(parent.topology.index_map(2).size_local, dtype=np.int64)
    region_of_cell[old.cell_tags.indices] = old.cell_tags.values
    region = np.array(
        [
            region_of_cell[hits.links(np.int32(i))[0]] - 1 if len(hits.links(np.int32(i))) else -1
            for i in range(len(probe))
        ]
    )
    left = boundary.pairs[:, 1]
    assert np.array_equal(np.where(left < 0, -1, left), region)


def test_a_remesh_of_a_moved_stretched_partition_adds_points_only_on_the_old_segments() -> None:
    # Stretch the disk 2.5x (r = 0.3 -> 0.75), compressing the annulus out to r = 0.95 and leaving the rest, a
    # monotone radial map, so its membrane's segments grow well past h and Netgen adds points on them; every mesh
    # (the parent and each submesh) takes the same map at its coincident nodes.
    description = _box(
        "disk",
        (
            SubVolume(name="cyto", type="analytic", expression=f"{_R2} < 0.09"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        (SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )
    old = realize_multi_compartment(description, h=0.08)

    def stretch(x: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(x[:, :2], axis=1)
        f = np.where(r <= 0.3, 2.5 * r, np.where(r <= 0.95, 0.75 + (r - 0.3) * 0.2 / 0.65, r))
        out = x.copy()
        out[:, :2] *= (f / np.where(r > 0.0, r, 1.0))[:, None]
        return out

    for mesh in [
        old.parent_mesh,
        *(c.mesh for c in old.compartments.values()),
        *(m.mesh for m in old.membranes.values()),
    ]:
        mesh.geometry.x[:] = stretch(mesh.geometry.x)
    new = remesh_partition(old, description, h=0.08)
    n_old = old.membranes["pm"].mesh.geometry.x.shape[0]
    n_new = new.membranes["pm"].mesh.geometry.x.shape[0]
    assert n_new > n_old  # the stretched segments were split, exercising the added points
    _check_remesh(old, new)


def test_a_steep_field_takes_the_lumped_transfer_and_stays_non_negative() -> None:
    description = _nested()
    old = realize_multi_compartment(description, h=0.1)
    new = remesh_partition(old, description, h=0.1)
    V_old = fem.functionspace(old.compartments["cyto"].mesh, ("Lagrange", 1))
    V_new = fem.functionspace(new.compartments["cyto"].mesh, ("Lagrange", 1))
    step = fem.Function(V_old)
    step.interpolate(lambda x: np.where(x[0] < 0.05, 1.0, 0.0))
    moved, kinds = transfer_compartment({"step": step}, V_new)
    assert kinds["step"] == "lumped"
    assert moved["step"].x.array.min() >= 0.0


def test_a_total_that_changes_stops_the_run() -> None:
    _check_total("u", (1.0, 1.0), (1.0 + 5e-13, 1.0))  # within round-off
    with pytest.raises(RemeshLostMass, match="changed the total of 'u'"):
        _check_total("u", (1.0, 1.0), (1.0 + 1e-9, 1.0))


def test_a_skeleton_with_its_sides_swapped_is_refused_before_netgen_sees_it() -> None:
    # Netgen hangs on an inconsistently oriented skeleton (found while testing this), so the remesh checks that
    # every region's oriented boundary encloses exactly its cells' area
    from vcell_fenics.backend.label_surfaces import LabelBoundary
    from vcell_fenics.backend.partition_remesh import PartitionRemeshError, _check_orientation

    old = realize_multi_compartment(_nested(), h=0.15)
    boundary = skeleton(old)
    parent = old.parent_mesh
    corners = parent.geometry.x[np.asarray(parent.geometry.dofmap)][:, :, :2]
    u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])  # not np.cross: NumPy 2.5 dropped 2D vectors
    region = np.zeros(len(area), dtype=np.int64)
    region[old.cell_tags.indices] = old.cell_tags.values
    _check_orientation(boundary, region - 1, area)  # as built: consistent
    swapped = LabelBoundary(dim=2, points=boundary.points, elements=boundary.elements, pairs=boundary.pairs[:, ::-1])
    with pytest.raises(PartitionRemeshError, match="skeleton is inconsistent"):
        _check_orientation(swapped, region - 1, area)


def test_a_smooth_field_that_undershoots_but_stays_positive_keeps_the_projection() -> None:
    # ADR 016 decision 3 as amended: the projection undershoots a smooth field's minimum by its interpolation
    # error, which is no reason to fall back; only a field it would make negative does. A field already negative
    # somewhere keeps the projection too.
    description = _nested()
    old = realize_multi_compartment(description, h=0.12)
    new = remesh_partition(old, description, h=0.12)
    V_old = fem.functionspace(old.compartments["ext"].mesh, ("Lagrange", 1))
    V_new = fem.functionspace(new.compartments["ext"].mesh, ("Lagrange", 1))
    smooth, signed = fem.Function(V_old), fem.Function(V_old)
    smooth.interpolate(lambda x: 1.0 + np.cos(3.0 * x[0]))
    signed.interpolate(lambda x: np.sin(4.0 * x[0]))
    moved, kinds = transfer_compartment({"smooth": smooth, "signed": signed}, V_new)
    assert moved["smooth"].x.array.min() < smooth.x.array.min()  # it does undershoot the old minimum
    assert moved["smooth"].x.array.min() > 0.0  # but stays positive
    assert kinds == {"smooth": "projection", "signed": "projection"}
