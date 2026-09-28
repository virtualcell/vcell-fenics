"""The label field of an image-based geometry: which subvolume owns each point of a lattice.

The first stage of image realization (`geometric-formalism.md` §3.2). A VCell image geometry is a
segmented uint8 image whose pixel values are subvolumes; this module turns it into a **label grid**
— an integer array of subvolume indices on a regular lattice — in three steps:

1. **The image's own lattice** (:func:`image_label_grid`). VCell's lattice is *vertex-centred*: pixel
   ``i`` sits at ``origin + i·extent / (n − 1)``, so the first and last pixels lie on the domain
   boundary. Analytic subvolumes, which VCell allows alongside image ones, are rasterized on top:
   they win where their predicate holds, the earliest in subvolume order first (VCell's priority).
2. **Smoothing and resampling** (:func:`smoothed_label_grid`). Each subvolume's indicator is
   Gaussian-smoothed (``σ`` in pixels, per axis, so anisotropic voxels are smoothed by their own
   spacing), resampled trilinearly onto a lattice of spacing ≈ ``h`` whose nodes include the box
   faces, and each node takes the subvolume of the largest smoothed indicator. This removes the
   pixel staircase deterministically and ties the surface resolution to the mesh size.
3. **Clean-up.** Specks smoothing severed from a thin protrusion are absorbed (:func:`drop_fragments`),
   and pinches repaired (:func:`repair_pinches`): two cells of one subvolume touching only at an edge
   or a corner would make the extracted surface non-manifold there, so one of them is reassigned.

:func:`label_geometry` runs all three and reports what smoothing changed (a subvolume that vanishes
is an error; one whose connected pieces merge or split is a warning), plus membranes whose
subvolume pair has no ``SurfaceClass``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from vcell_fenics.backend.implicit_fields import RealizationError, eval_field
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import lower_predicate, subvolume_implicit_functions

Labels = NDArray[np.int32]

# a piece smoothing severed is a speck (absorbed) when smaller than this fraction of its subvolume
_SPECK_FRACTION = 0.02


@dataclass(frozen=True)
class LabelGrid:
    """Subvolume indices (into ``GeometryDescription.subvolumes``) on a regular lattice.

    ``labels`` is indexed ``[x, y]`` in 2D and ``[x, y, z]`` in 3D; node ``i`` along an axis sits at
    ``origin + i·spacing`` (so the first and last nodes lie on the box faces)."""

    labels: Labels
    origin: tuple[float, ...]
    spacing: tuple[float, ...]
    # the smoothed indicator of each subvolume on this lattice (``None`` for one absent from the image),
    # when the grid came from smoothing: continuous, so the boundary between a and b is I_a = I_b
    indicators: tuple[NDArray[np.float32] | None, ...] | None = None
    # the indicators as exact functions of points (n, dim), when known (an analytic geometry's): the
    # boundary projection then lands on the true interface, not its trilinear interpolant
    exact: tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], ...] | None = None

    @property
    def dim(self) -> int:
        return self.labels.ndim

    def axis(self, i: int) -> NDArray[np.float64]:
        """The node coordinates along axis ``i``."""

        return self.origin[i] + self.spacing[i] * np.arange(self.labels.shape[i], dtype=np.float64)


def image_label_grid(description: GeometryDescription) -> LabelGrid:
    """The label grid on the image's own vertex-centred lattice, analytic subvolumes rasterized on
    top. Raises :class:`RealizationError` if the geometry has no image voxels, or a pixel value that
    no subvolume maps."""

    image = description.image
    if image is None or image.compressed_content is None:
        raise RealizationError(f"geometry {description.name!r} has image subvolumes but no image voxels to mesh")
    dim = description.dim
    voxels = image.voxels()  # (nz, ny, nx)
    if dim == 2:
        if voxels.shape[0] != 1:
            raise RealizationError(f"2D geometry {description.name!r} has a {voxels.shape[0]}-slice image")
        pixels = voxels[0].T  # → [x, y]
    elif dim == 3:
        pixels = voxels.transpose(2, 1, 0)  # → [x, y, z]
    else:
        raise RealizationError(f"image geometry {description.name!r} must be 2D or 3D (dim {dim})")
    if min(pixels.shape) < 2:
        raise RealizationError(f"image {image.name!r} needs at least 2 pixels along each axis, got {pixels.shape}")

    lookup = np.full(256, -1, dtype=np.int32)
    for index, subvolume in enumerate(description.subvolumes):
        if subvolume.type == "image" and subvolume.pixel_value is not None:
            lookup[subvolume.pixel_value] = index
    labels = lookup[pixels]
    if bool((labels < 0).any()):
        unmapped = sorted({int(v) for v in np.unique(pixels[labels < 0])})
        raise RealizationError(f"image {image.name!r} has pixel values {unmapped} that no subvolume maps")

    origin = tuple(float(description.origin[i]) for i in range(dim))
    spacing = tuple(float(description.extent[i]) / (pixels.shape[i] - 1) for i in range(dim))
    grid = LabelGrid(labels=labels, origin=origin, spacing=spacing)
    _overlay_analytic(description, grid)
    return grid


def _overlay_analytic(description: GeometryDescription, grid: LabelGrid) -> None:
    """Rasterize analytic subvolumes onto ``grid`` in place: each owns the nodes where its predicate
    holds, the earliest subvolume winning where several do (applied in reverse order)."""

    analytic = [(i, s) for i, s in enumerate(description.subvolumes) if s.type == "analytic" and s.expression]
    if not analytic:
        return
    coords = np.meshgrid(*(grid.axis(i) for i in range(grid.dim)), indexing="ij")
    for index, subvolume in reversed(analytic):
        assert subvolume.expression is not None
        phi = eval_field(lower_predicate(parse(subvolume.expression)), tuple(coords))
        grid.labels[phi <= 0.0] = index


def smoothed_label_grid(raw: LabelGrid, *, extent: tuple[float, ...], h: float, sigma_pixels: float = 1.0) -> LabelGrid:
    """Smooth ``raw`` and resample it onto a lattice of spacing ≈ ``h`` spanning ``extent`` (its nodes
    include the box faces), each node taking the subvolume of largest smoothed indicator. Pinches are
    repaired (:func:`repair_pinches`)."""

    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    dim = raw.dim
    counts = tuple(max(3, round(extent[i] / h) + 1) for i in range(dim))
    spacing = tuple(extent[i] / (counts[i] - 1) for i in range(dim))
    # the target nodes in the raw lattice's (fractional) pixel coordinates
    pixel_coords = np.meshgrid(
        *(np.linspace(0.0, raw.labels.shape[i] - 1, counts[i]) for i in range(dim)), indexing="ij"
    )
    present = np.unique(raw.labels)
    best = np.full(counts, -np.inf, dtype=np.float32)
    labels = np.zeros(counts, dtype=np.int32)
    indicators: list[NDArray[np.float32] | None] = [None] * (int(raw.labels.max()) + 1)
    for value in present:
        indicator = ndimage.gaussian_filter((raw.labels == value).astype(np.float32), sigma_pixels, mode="nearest")
        sampled = ndimage.map_coordinates(indicator, pixel_coords, order=1, mode="nearest").astype(np.float32)
        indicators[int(value)] = sampled
        wins = sampled > best
        best[wins] = sampled[wins]
        labels[wins] = value
    labels = drop_fragments(labels, {int(v): int(ndimage.label(raw.labels == v)[1]) for v in present})
    return LabelGrid(labels=repair_pinches(labels), origin=raw.origin, spacing=spacing, indicators=tuple(indicators))


def drop_fragments(labels: Labels, pieces: dict[int, int]) -> Labels:
    """``labels`` with the specks smoothing made removed: smoothing can sever a thin protrusion into a
    tiny piece the image never had. Of a subvolume with more pieces than ``pieces[value]`` (its count in
    the image), the smallest excess pieces are absorbed — but only those under ``_SPECK_FRACTION`` of the
    subvolume's size, so a large piece severed by smoothing (a dumbbell cut at its neck) is kept, and
    reported as a split. An absorbed speck takes the most common label around it."""

    out = labels.copy()
    for value, keep in pieces.items():
        components, n = ndimage.label(out == value)
        if n <= keep:
            continue
        sizes = np.bincount(components.ravel())[1:]
        limit = _SPECK_FRACTION * float(sizes.sum())
        for piece in np.argsort(sizes)[: n - keep] + 1:  # the smallest n − keep pieces
            if sizes[piece - 1] >= limit:
                continue
            speck = components == piece
            ring = ndimage.binary_dilation(speck) & ~speck
            around = out[ring]
            values, counts = np.unique(around[around != value], return_counts=True)
            if values.size:
                out[speck] = int(values[np.argmax(counts)])
    return out


def find_pinches(labels: Labels) -> NDArray[np.bool_]:
    """A mask over ``labels`` of the cells that sit in a pinch: two cells of one subvolume touching
    only diagonally — across a square's diagonal (2D; and in each axis plane of a 3D lattice) or a
    cube's (3D), with no face-connected path between them inside that square or cube."""

    mask = np.zeros(labels.shape, dtype=bool)
    for a, b in _axis_pairs(labels.ndim):
        _mark_square_pinches(labels, a, b, mask)
    if labels.ndim == 3:
        _mark_cube_pinches(labels, mask)
    return mask


def repair_pinches(labels: Labels, *, max_passes: int = 50) -> Labels:
    """``labels`` with every pinch removed. Each pinched cell is reassigned to whichever label present
    around it leaves the fewest pinches in its neighbourhood (ties: the most common neighbour) — a greedy
    local choice, so a configuration cannot flip back and forth (a fixed "take a face neighbour's label"
    rule 2-cycled on a thin cytosol). One cell per neighbourhood per pass: both diagonals of a
    checkerboard square are pinches, and flipping both at once would just invert it. Raises
    :class:`RealizationError` if pinches persist after ``max_passes``."""

    out = labels.copy()
    for _ in range(max_passes):
        mask = find_pinches(out)
        if not mask.any():
            return out
        touched = np.zeros(out.shape, dtype=bool)
        for index in zip(*np.nonzero(mask), strict=True):
            near = tuple(slice(max(0, i - 1), i + 2) for i in index)
            if touched[near].any():
                continue
            touched[index] = True
            cell, value = _best_move(out, index)
            out[cell] = value
    if find_pinches(out).any():
        raise RealizationError(f"could not repair diagonal pinches in the label field after {max_passes} passes")
    return out


def _best_move(labels: Labels, index: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """The single relabelling near the pinched cell at ``index`` — of the cell itself or of one of its
    neighbours, to a label present around it — that leaves the fewest pinches in the window around it
    (every 2-wide block any such cell belongs to lies inside). Ties prefer changing the pinched cell
    itself, then the most common label. Searching the neighbours too matters: a pinch can be one no
    relabelling of its own cell removes."""

    dim = labels.ndim
    window = tuple(slice(max(0, i - 3), i + 4) for i in index)
    start = tuple(w.start for w in window)
    around = labels[tuple(slice(max(0, i - 1), i + 2) for i in index)]
    values, counts = np.unique(around, return_counts=True)
    common = dict(zip(values.tolist(), counts.tolist(), strict=True))
    best: tuple[tuple[int, ...], int] = (index, int(labels[index]))
    best_key: tuple[float, int, int] = (np.inf, 1, 0)
    for offset in np.ndindex(*(3,) * dim):
        cell = tuple(index[k] + offset[k] - 1 for k in range(dim))
        if any(c < 0 or c >= labels.shape[k] for k, c in enumerate(cell)):
            continue
        own = int(labels[cell])
        local = tuple(c - s for c, s in zip(cell, start, strict=True))
        for value in common:
            if value == own:
                continue
            trial = labels[window].copy()
            trial[local] = value
            key = (float(find_pinches(trial).sum()), 0 if cell == index else 1, -common[value])
            if key < best_key:
                best, best_key = (cell, value), key
    return best


def _axis_pairs(dim: int) -> list[tuple[int, int]]:
    return [(0, 1)] if dim == 2 else [(0, 1), (0, 2), (1, 2)]


def _face_steps(dim: int) -> list[tuple[int, ...]]:
    steps: list[tuple[int, ...]] = []
    for k in range(dim):
        for sign in (-1, 1):
            step = [0] * dim
            step[k] = sign
            steps.append(tuple(step))
    return steps


def _corner(labels: Labels, offsets: dict[int, int]) -> tuple[slice, ...]:
    """The slice selecting, for every 2-wide block, the corner at ``offsets`` (axis → 0/1)."""

    return tuple(slice(offsets.get(k, 0), labels.shape[k] - 1 + offsets.get(k, 0)) for k in range(labels.ndim))


def _mark_square_pinches(labels: Labels, a: int, b: int, mask: NDArray[np.bool_]) -> None:
    """Mark pinches across the diagonals of the ``(a, b)``-plane squares (the far cell of each)."""

    shape = [s - 1 if k in (a, b) else s for k, s in enumerate(labels.shape)]

    def block(da: int, db: int) -> tuple[slice, ...]:
        return tuple(
            slice(da, da + shape[k]) if k == a else slice(db, db + shape[k]) if k == b else slice(0, shape[k])
            for k in range(labels.ndim)
        )

    c00, c10, c01, c11 = labels[block(0, 0)], labels[block(1, 0)], labels[block(0, 1)], labels[block(1, 1)]
    main = (c00 == c11) & (c10 != c00) & (c01 != c00)
    anti = (c10 == c01) & (c00 != c10) & (c11 != c10)
    mask[block(1, 1)] |= main
    mask[block(0, 1)] |= anti


def _mark_cube_pinches(labels: Labels, mask: NDArray[np.bool_]) -> None:
    """Mark corner-only contacts across a cube's four main diagonals (the far cell of each): the two
    ends share a label and none of the cube's other six cells does (a partial path is an edge pinch,
    which the square check already catches)."""

    corners = {(i, j, k): labels[_corner(labels, {0: i, 1: j, 2: k})] for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for near in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        far: tuple[int, int, int] = (1 - near[0], 1 - near[1], 1 - near[2])
        ends = corners[near] == corners[far]
        for other, values in corners.items():
            if other not in (near, far):
                ends &= values != corners[near]
        mask[_corner(labels, {0: far[0], 1: far[1], 2: far[2]})] |= ends


def component_counts(labels: Labels, n_subvolumes: int) -> list[int]:
    """The number of face-connected pieces of each subvolume in ``labels``."""

    return [int(ndimage.label(labels == index)[1]) for index in range(n_subvolumes)]


def adjacent_pairs(labels: Labels) -> set[tuple[int, int]]:
    """The unordered subvolume pairs that share a face somewhere in ``labels``."""

    pairs: set[tuple[int, int]] = set()
    for k in range(labels.ndim):
        low = np.take(labels, range(labels.shape[k] - 1), axis=k).ravel()
        high = np.take(labels, range(1, labels.shape[k]), axis=k).ravel()
        differ = low != high
        for p, q in np.unique(np.stack([low[differ], high[differ]], axis=1), axis=0) if differ.any() else ():
            pairs.add((int(min(p, q)), int(max(p, q))))
    return pairs


@dataclass(frozen=True)
class LabelGeometry:
    """The realized label field of an image geometry and what building it reported."""

    grid: LabelGrid
    components: tuple[int, ...]
    warnings: tuple[str, ...]


def label_geometry(description: GeometryDescription, *, h: float, sigma_pixels: float = 1.0) -> LabelGeometry:
    """The smoothed label field of an image geometry at mesh size ≈ ``h`` (see the module docstring).

    Raises :class:`RealizationError` if a subvolume present in the image vanishes under smoothing at
    this ``h`` (refine ``h``); warns when a subvolume's connected pieces merge or split, and for each
    touching subvolume pair without a ``SurfaceClass`` (that interface gets no membrane)."""

    raw = image_label_grid(description)
    dim = description.dim
    smooth = smoothed_label_grid(
        raw, extent=tuple(float(description.extent[i]) for i in range(dim)), h=h, sigma_pixels=sigma_pixels
    )
    names = [s.name for s in description.subvolumes]
    before = component_counts(raw.labels, len(names))
    after = component_counts(smooth.labels, len(names))
    warnings: list[str] = []
    vanished = [names[i] for i in range(len(names)) if before[i] > 0 and after[i] == 0]
    if vanished:
        raise RealizationError(
            f"image geometry {description.name!r}: subvolume(s) {vanished} vanish at mesh size h = {h:g} "
            f"(thinner than the smoothing); use a smaller h"
        )
    for i, name in enumerate(names):
        if before[i] != after[i]:
            warnings.append(
                f"subvolume {name!r} has {after[i]} connected piece(s) at h = {h:g} ({before[i]} in the image)"
            )
    warnings.extend(_undeclared_membranes(description, smooth.labels))
    return LabelGeometry(grid=smooth, components=tuple(after), warnings=tuple(warnings))


def _undeclared_membranes(description: GeometryDescription, labels: Labels) -> list[str]:
    """A warning for each touching subvolume pair that no ``SurfaceClass`` names (it gets no membrane)."""

    names = [s.name for s in description.subvolumes]
    declared = {
        tuple(sorted((names.index(s.inside), names.index(s.outside))))
        for s in description.surfaces
        if s.inside in names and s.outside in names
    }
    return [
        f"subvolumes {names[p]!r} and {names[q]!r} touch but no surface class names that membrane"
        for p, q in sorted(adjacent_pairs(labels) - declared)
    ]


def analytic_label_geometry(description: GeometryDescription, *, h: float) -> LabelGeometry:
    """The label field of an analytic geometry at mesh size ≈ ``h``, for the partitions the contour /
    marching path can't body-fit — a subvolume that touches the box (#187). No smoothing: each node of a
    lattice of spacing ≈ ``h`` whose nodes include the box faces takes the subvolume VCell's priority rule
    gives it, and the indicators are the priority-resolved implicit functions themselves (``I_k = −φ_k``,
    so the interface between two subvolumes, ``I_a = I_b``, is the analytic surface): the boundary
    extraction then projects its vertices onto the exact shapes, not onto a smoothed staircase.

    Raises :class:`RealizationError` for a subvolume thinner than the lattice (present on a 4× finer one,
    absent at ``h``)."""

    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    dim = description.dim
    extent = tuple(float(description.extent[i]) for i in range(dim))
    origin = tuple(float(description.origin[i]) for i in range(dim))
    fields = subvolume_implicit_functions(description)
    names = [s.name for s in description.subvolumes]

    def lattice(step: float) -> tuple[tuple[float, ...], list[NDArray[np.float32]]]:
        counts = tuple(max(3, round(extent[i] / step) + 1) for i in range(dim))
        spacing = tuple(extent[i] / (counts[i] - 1) for i in range(dim))
        axes = [origin[i] + spacing[i] * np.arange(counts[i], dtype=np.float64) for i in range(dim)]
        coords = tuple(np.meshgrid(*axes, indexing="ij"))
        return spacing, [(-eval_field(fields[name], coords)).astype(np.float32) for name in names]

    spacing, indicators = lattice(h)
    # the owner is the subvolume whose resolved function is inside (φ ≤ 0): the largest indicator, the
    # earliest (highest-priority) subvolume on a tie
    labels = np.argmax(np.stack(indicators), axis=0).astype(np.int32)
    labels = repair_pinches(labels)
    counts = component_counts(labels, len(names))
    missing = [i for i in range(len(names)) if counts[i] == 0]
    if missing:
        _, fine = lattice(h / 4.0)
        owners = set(np.unique(np.argmax(np.stack(fine), axis=0)).tolist())
        thin = [names[i] for i in missing if i in owners]
        if thin:
            raise RealizationError(
                f"geometry {description.name!r}: subvolume(s) {thin} are thinner than the mesh size h = {h:g}; "
                "use a smaller h"
            )

    def exact(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        field = fields[name]
        return lambda points: -eval_field(field, tuple(points[:, i] for i in range(dim)))

    grid = LabelGrid(
        labels=labels,
        origin=origin,
        spacing=spacing,
        indicators=tuple(indicators),
        exact=tuple(exact(name) for name in names),
    )
    return LabelGeometry(
        grid=grid, components=tuple(counts), warnings=tuple(_undeclared_membranes(description, labels))
    )
