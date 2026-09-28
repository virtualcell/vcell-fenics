"""Conforming boundaries of a label field: the curves (2D) or surfaces (3D) between its subvolumes.

The second stage of image realization (`geometric-formalism.md` §3.2), after :mod:`labels`. A
:class:`~vcell_fenics.backend.labels.LabelGrid` is turned into a :class:`LabelBoundary` — oriented
segments (2D) or triangles (3D), each carrying the pair of regions on its two sides — that the
mesher (``realize.py``) embeds as conforming internal boundaries:

- **Extraction** is VTK's SurfaceNets (``vtkSurfaceNets2D`` / ``3D``), which meshes *every* pair of
  touching labels at once, so boundaries meet conformingly where three or more subvolumes meet (a
  junction curve or point) — which per-subvolume marching cannot do.
- **The box.** The grid is padded with one **sentinel label per box face**, so a subvolume touching
  the box gets its box-face boundary too, and box edges and corners become ordinary junctions. A
  sentinel side is recorded as ``-(k + 1)`` for face ``k`` (``x_minus, x_plus, y_minus, y_plus,
  z_minus, z_plus``); those vertices are projected exactly onto their face planes.
- **Smoothing** is our own, deterministic and constrained (VTK's shrinks small regions): Taubin
  passes (λ, μ) move a vertex on one interface freely, a vertex on a box face within its plane, a
  vertex on a junction curve or box edge only along that curve, and never a vertex where four or
  more sides meet (curve ends, box corners, 2D junction points).
- **Orientation.** Each element's normal points from ``pairs[:, 0]`` into ``pairs[:, 1]`` — in 3D
  the triangle normal (right-hand rule), in 2D the left normal of segment ``p → q``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from vtkmodules.util import numpy_support
from vtkmodules.util.vtkConstants import VTK_INT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkSurfaceNets2D, vtkSurfaceNets3D

from vcell_fenics.backend.implicit_fields import RealizationError
from vcell_fenics.backend.labels import LabelGrid

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

_TAUBIN = (0.5, -0.53)


@dataclass(frozen=True)
class LabelBoundary:
    """The boundaries of a label field. ``elements`` are segments (2D) or triangles (3D) into
    ``points``; ``pairs[e] = (a, b)`` are the sides of element ``e`` — a subvolume index, or
    ``-(k + 1)`` for box face ``k`` — with the element's normal pointing from ``a`` into ``b``."""

    dim: int
    points: Floats
    elements: Ints
    pairs: Ints

    def measure(self) -> Floats:
        """Each element's length (2D) or area (3D)."""

        corners = self.points[self.elements]
        if self.dim == 2:
            return np.asarray(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1), dtype=np.float64)
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        return np.asarray(0.5 * np.linalg.norm(cross, axis=1), dtype=np.float64)

    def region_measure(self, region: int) -> float:
        """The area (2D) or volume (3D) enclosed by ``region``'s oriented boundary (the divergence
        theorem: exact for a closed boundary, so the pieces of a partition sum to the box)."""

        corners = self.points[self.elements]
        if self.dim == 2:
            p, q = corners[:, 0], corners[:, 1]
            flux = 0.5 * (p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])  # ∮ (x dy − y dx)/2, positive for the left side
        else:
            flux = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])) / 6.0
        # the normal points into pairs[:, 1]: outward for pairs[:, 0]
        outward = (self.pairs[:, 0] == region).astype(np.float64) - (self.pairs[:, 1] == region)
        # 2D: the left normal points into pairs[:, 1], so the left side is pairs[:, 1]
        sign = -1.0 if self.dim == 2 else 1.0
        return float(sign * np.sum(flux * outward))


def face_names(dim: int) -> tuple[str, ...]:
    return ("x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus")[: 2 * dim]


def extract_boundary(
    grid: LabelGrid, *, extent: tuple[float, ...], passes: int = 10, project: bool = True
) -> LabelBoundary:
    """The conforming, oriented, smoothed boundaries of ``grid`` (see the module docstring). ``extent``
    is the box size (the grid spans ``[origin, origin + extent]``); ``passes`` Taubin pairs smooth it, and
    ``project`` moves the vertices onto the smooth interfaces (when the grid carries its indicators).
    With ``passes = 0`` and no projection the boundary is SurfaceNets' own — one vertex per lattice
    cell, so it cannot intersect itself: the mesher's last resort."""

    dim = grid.dim
    n_regions = int(grid.labels.max()) + 1
    points, elements, vtk_pairs = _surface_nets(grid, n_regions)
    pairs = np.where(vtk_pairs <= n_regions, vtk_pairs - 1, -(vtk_pairs - n_regions))
    keep = (pairs[:, 0] >= 0) | (pairs[:, 1] >= 0)  # drop sentinel–sentinel elements
    elements, pairs = elements[keep], pairs[keep]
    used, elements = np.unique(elements, return_inverse=True)
    elements = elements.reshape(-1, dim)
    points = points[used]
    if dim == 2:
        elements, pairs = _orient_2d(points, elements, pairs, grid)
    else:
        elements, pairs = _sided_3d(elements, pairs)
    lo = np.asarray(grid.origin[:dim], dtype=np.float64)
    hi = lo + np.asarray(extent[:dim], dtype=np.float64)
    boundary = LabelBoundary(dim=dim, points=points, elements=elements, pairs=pairs)
    return _smooth(boundary, lo, hi, passes, grid if project else replace(grid, indicators=None))


def _surface_nets(grid: LabelGrid, n_regions: int) -> tuple[Floats, Ints, Ints]:
    """Run SurfaceNets on the sentinel-padded grid. Returns points, elements and the (sorted) VTK label
    pair of each element: region ``r`` is label ``r + 1``, box face ``k`` label ``n_regions + 1 + k``."""

    dim = grid.dim
    padded = np.pad(grid.labels.astype(np.int32) + 1, 1)
    for axis in range(dim):
        for side, index in ((0, 0), (1, -1)):
            face = [slice(None)] * dim
            face[axis] = slice(index, None) if index == -1 else slice(0, 1)
            padded[tuple(face)] = n_regions + 1 + 2 * axis + side
    image = vtkImageData()
    shape = padded.shape if dim == 3 else (*padded.shape, 1)
    image.SetDimensions(*shape)
    spacing = (*grid.spacing, 1.0) if dim == 2 else grid.spacing
    origin = tuple(grid.origin[i] - grid.spacing[i] for i in range(dim))
    image.SetSpacing(*spacing)
    image.SetOrigin(*(origin if dim == 3 else (*origin, 0.0)))
    scalars = numpy_support.numpy_to_vtk(padded.ravel(order="F"), deep=True, array_type=VTK_INT)
    image.GetPointData().SetScalars(scalars)

    nets: vtkSurfaceNets2D | vtkSurfaceNets3D
    if dim == 2:
        nets = vtkSurfaceNets2D()
    else:
        nets = vtkSurfaceNets3D()
        nets.SetOutputMeshTypeToTriangles()
    nets.SetInputData(image)
    labels = sorted(int(v) for v in np.unique(padded))
    nets.SetNumberOfLabels(len(labels))
    for i, value in enumerate(labels):
        nets.SetLabel(i, value)
    nets.SetBackgroundLabel(0)  # no node carries 0 after the shift, so every interface is output
    nets.SetSmoothing(False)
    nets.Update()
    output = nets.GetOutput()
    if output.GetNumberOfPoints() == 0:
        raise RealizationError("the label field has no internal boundaries to mesh")
    points = numpy_support.vtk_to_numpy(output.GetPoints().GetData()).astype(np.float64)[:, :dim].copy()
    cells = output.GetLines() if dim == 2 else output.GetPolys()
    elements = numpy_support.vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64).reshape(-1, dim)
    pairs = numpy_support.vtk_to_numpy(output.GetCellData().GetArray("BoundaryLabels")).astype(np.int64)
    return points, elements, pairs.reshape(-1, 2)


def _sided_3d(elements: Ints, pairs: Ints) -> tuple[Ints, Ints]:
    """VTK orients each triangle's normal toward its second label; keep that, as ``(a, b)`` in the
    same order."""

    return elements, pairs


def _orient_2d(points: Floats, elements: Ints, pairs: Ints, grid: LabelGrid) -> tuple[Ints, Ints]:
    """Orient each segment so its left normal points into ``pairs[:, 1]``. SurfaceNets2D emits its
    segments in no fixed direction; before smoothing a segment separates two grid nodes, so the node
    half a cell along its left normal names its left side exactly."""

    p, q = points[elements[:, 0]], points[elements[:, 1]]
    direction = q - p
    left = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    probe = 0.5 * (p + q) + 0.5 * left * np.asarray(grid.spacing)
    index = np.rint((probe - np.asarray(grid.origin)) / np.asarray(grid.spacing)).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(grid.labels.shape)), axis=1)
    left_label = np.full(len(elements), -1, dtype=np.int64)
    left_label[inside] = grid.labels[index[inside, 0], index[inside, 1]]
    # the probe falls outside the grid only toward a box face: that side is the sentinel
    region_left = np.where(inside, left_label == pairs[:, 1], pairs[:, 1] < 0)
    flip = ~region_left
    elements = elements.copy()
    elements[flip] = elements[flip][:, ::-1]
    return elements, pairs


def _smooth(boundary: LabelBoundary, lo: Floats, hi: Floats, passes: int, grid: LabelGrid) -> LabelBoundary:
    """Snap box-face vertices onto their planes, smooth by vertex class (the module docstring), then —
    when the grid carries its smoothed indicators — project each vertex onto the smooth interface
    (:func:`_project`), so the boundary converges instead of keeping the lattice's roughness."""

    dim = boundary.dim
    points = boundary.points.copy()
    n = len(points)
    # each vertex's sides: the union of its elements' pairs
    incident = np.repeat(boundary.pairs, dim, axis=0)
    owner = boundary.elements.ravel()
    sides: list[set[int]] = [set() for _ in range(n)]
    for vertex, (a, b) in zip(owner.tolist(), incident.tolist(), strict=True):
        sides[vertex].add(a)
        sides[vertex].add(b)
    # box faces each vertex lies on → the axes it is pinned on, and the value there
    pinned = np.zeros((n, dim), dtype=bool)
    for vertex, labels in enumerate(sides):
        for label in labels:
            if label < 0:
                k = -label - 1
                axis, top = divmod(k, 2)
                pinned[vertex, axis] = True
                points[vertex, axis] = hi[axis] if top else lo[axis]
    counts = np.array([len(s) for s in sides])
    # movable: an interface (2 sides) anywhere, or a 3D curve (3 sides); fixed otherwise
    movable = (counts == 2) | ((counts == 3) & (dim == 3))
    edges = _edges(boundary.elements, dim)
    u, v = edges[:, 0], edges[:, 1]
    # a curve vertex only averages over neighbours on the same curve (the same set of sides)
    same = np.array([sides[a] == sides[b] for a, b in zip(u.tolist(), v.tolist(), strict=True)], dtype=bool)
    keep_uv = (counts[u] == 2) | same
    keep_vu = (counts[v] == 2) | same
    rows = np.concatenate([u[keep_uv], v[keep_vu]])
    cols = np.concatenate([v[keep_uv], u[keep_vu]])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # a curve vertex needs exactly two curve neighbours (an interior point of the curve)
    movable &= np.where(counts == 3, degree == 2, degree > 0)
    for _ in range(passes):
        for factor in _TAUBIN:
            mean = adjacency @ points / np.maximum(degree, 1.0)[:, None]
            step = factor * (mean - points)
            step[~movable] = 0.0
            step[pinned] = 0.0
            points += step
    if grid.indicators is not None:
        projected = _project(points, sides, pinned, grid)
        if grid.exact is not None:
            # Exact projection moves interface vertices by up to a cell, which can overtake the unprojected
            # vertices beside them (a box face's, next to where an interface meets it) and flip their
            # elements; let those follow — Laplacian passes with the projected vertices held.
            onto = np.array([2 <= len({label for label in labels if label >= 0}) <= 3 for labels in sides])
            follow = movable & ~onto
            for _ in range(passes):
                mean = adjacency @ projected / np.maximum(degree, 1.0)[:, None]
                step = 0.5 * (mean - projected)
                step[~follow] = 0.0
                step[pinned] = 0.0
                projected += step
        points = _guarded(points, projected, boundary.elements, grid)
    return LabelBoundary(dim=dim, points=points, elements=boundary.elements, pairs=boundary.pairs)


def _guarded(before: Floats, after: Floats, elements: Ints, grid: LabelGrid) -> Floats:
    """``after`` (the projected points), except where the projection is not to be trusted, which keep
    their ``before`` (smoothed) positions: a vertex whose target lies more than half a cell away (where
    clean-up relabelled nodes, the smoothed indicators no longer agree with the labels, so their level
    set can lie across the extracted surface), and the vertices of any element the projection turned
    over or twisted sharply (repeated until none is)."""

    # exact fields agree with the labels by construction, so only a jump past the neighbouring node is
    # suspect; a box-face vertex's true crossing can lie up to a cell from where SurfaceNets put it
    limit = (1.0 if grid.exact is not None else 0.5) * float(min(grid.spacing))
    out = after.copy()
    far = np.linalg.norm(after - before, axis=1) > limit
    out[far] = before[far]
    reference = _normals(before, elements)
    # rotated by > 60° — or, onto exact fields, inverted (> 90°): an exact interface meeting a box face at a
    # right angle legitimately turns the staircase's elements there by more than 60°
    threshold = 0.0 if grid.exact is not None else 0.5
    # onto exact fields a turned element's vertices back off halfway toward ``before`` each round (so they
    # keep as much of the exact projection as the element allows); otherwise they revert at once
    backoff = 0.5 if grid.exact is not None else 0.0
    fraction = np.ones(len(out))
    tiny = 1e-4 * float(min(grid.spacing)) ** (elements.shape[1] - 1)  # a collapsed element has no normal
    for round in range(10):
        turned = np.einsum("ij,ij->i", _normals(out, elements), reference) < threshold
        turned |= element_measures(out, elements) < tiny
        if not turned.any():
            break
        revert = np.unique(elements[turned])
        fraction[revert] = 0.0 if round == 9 else fraction[revert] * backoff
        out[revert] = before[revert] + fraction[revert, None] * (after[revert] - before[revert])
    return out


def element_measures(points: Floats, elements: Ints) -> Floats:
    """The elements' lengths (2D) or areas (3D)."""

    corners = points[elements]
    if elements.shape[1] == 2:
        return np.asarray(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1), dtype=np.float64)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.asarray(0.5 * np.linalg.norm(cross, axis=1), dtype=np.float64)


def _normals(points: Floats, elements: Ints) -> Floats:
    """Unit normals of the elements (the left normal of a segment in 2D)."""

    corners = points[elements]
    if elements.shape[1] == 2:
        direction = corners[:, 1] - corners[:, 0]
        normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    else:
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.asarray(normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300), dtype=np.float64)


def _project(points: Floats, sides: list[set[int]], pinned: NDArray[np.bool_], grid: LabelGrid) -> Floats:
    """Move each vertex onto the smooth interface(s) it lies on: for sides ``{a, b}`` the level set
    ``I_a = I_b`` of the smoothed indicators, for a junction ``{a, b, c}`` also ``I_b = I_c`` (a curve in
    3D, a point in 2D). Gauss–Newton on the trilinear interpolants, a few steps, each capped at half a
    cell; box-face axes stay pinned, and vertices with more than three region sides stay put."""

    assert grid.indicators is not None
    spacing = np.asarray(grid.spacing)
    origin = np.asarray(grid.origin)
    top = origin + spacing * (np.asarray(grid.labels.shape) - 1)
    inner_lo, inner_hi = origin + 0.25 * spacing, top - 0.25 * spacing
    out = points.copy()
    by_sides: dict[tuple[int, ...], list[int]] = {}
    for vertex, labels in enumerate(sides):
        regions = tuple(sorted(label for label in labels if label >= 0))
        if 2 <= len(regions) <= 3:
            by_sides.setdefault(regions, []).append(vertex)
    for regions, vertices in by_sides.items():
        fields = [grid.indicators[r] for r in regions]
        if any(f is None for f in fields):
            continue
        exact = [grid.exact[r] for r in regions] if grid.exact is not None else None
        idx = np.asarray(vertices)
        x = out[idx]
        free = ~pinned[idx]
        for _ in range(4):
            if exact is not None:
                values, gradients = zip(*(_sample_exact(f, x, spacing) for f in exact), strict=True)
            else:
                values, gradients = zip(*(_sample(f, x, origin, spacing) for f in fields if f is not None), strict=True)
            # constraints g_k = I_k − I_{k+1}, Jacobian rows ∇g_k (pinned axes held)
            g = np.stack([values[k] - values[k + 1] for k in range(len(fields) - 1)], axis=1)
            jac = np.stack([gradients[k] - gradients[k + 1] for k in range(len(fields) - 1)], axis=1)
            jac = jac * free[:, None, :]
            normal = jac @ np.swapaxes(jac, 1, 2) + 1e-12 * np.eye(len(fields) - 1)
            step = -np.einsum("vkd,vk->vd", jac, np.linalg.solve(normal, g[..., None])[..., 0])
            limit = 0.5 * float(spacing.min())
            length = np.linalg.norm(step, axis=1, keepdims=True)
            x = x + step * np.minimum(1.0, limit / np.maximum(length, 1e-300))
            if exact is not None:
                # an interface between two subvolumes never lies on a box face (that boundary is one
                # subvolume's and the outside's), but an exact predicate's zero set can (``z >= z0`` with
                # z0 the box's own face): keep free axes a quarter cell inside, or the vertex collapses there
                x = np.where(free, np.clip(x, inner_lo, inner_hi), x)
        out[idx] = x
    return out


def _sample(field: NDArray[np.float32], x: Floats, origin: Floats, spacing: Floats) -> tuple[Floats, Floats]:
    """A lattice field's trilinear interpolant and its gradient (central differences) at points ``x``."""

    from scipy import ndimage

    dim = x.shape[1]

    def at(points: Floats) -> Floats:
        coords = ((points - origin) / spacing).T
        return np.asarray(ndimage.map_coordinates(field, coords, order=1, mode="nearest"), dtype=np.float64)

    value = at(x)
    gradient = np.empty_like(x)
    for axis in range(dim):
        delta = np.zeros(dim)
        delta[axis] = 0.25 * spacing[axis]
        gradient[:, axis] = (at(x + delta) - at(x - delta)) / (2.0 * delta[axis])
    return value, gradient


def _sample_exact(field: Callable[[Floats], Floats], x: Floats, spacing: Floats) -> tuple[Floats, Floats]:
    """An exact field's value and gradient (central differences at a small fraction of a cell) at ``x``."""

    value = np.asarray(field(x), dtype=np.float64)
    gradient = np.empty_like(x)
    for axis in range(x.shape[1]):
        delta = np.zeros(x.shape[1])
        delta[axis] = 1e-3 * spacing[axis]
        gradient[:, axis] = (field(x + delta) - field(x - delta)) / (2.0 * delta[axis])
    return value, gradient


def _edges(elements: Ints, dim: int) -> Ints:
    """The unique undirected edges of the elements."""

    edges = elements if dim == 2 else np.concatenate([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.asarray(np.unique(edges, axis=0), dtype=np.int64)
