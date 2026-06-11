"""Conservative surface-density remap on a moving / remeshed membrane (1D case).

Implements the supermesh / Galerkin-conservative remap sketched in
`docs/modeling/conservative-surface-remap.md`: given a surface density ρ as P1
nodal values on one discretization of a closed membrane, transfer it to another
discretization of the same curve while preserving total surface mass ∫_Γ ρ ds
*exactly* (to linear-solve round-off).

Why this and not the mbsolver's polygon-overlap trick: two area control volumes
that occupy the same region literally intersect, so a clipping library hands you
the overlap area. Two nearby *polylines* meet only at isolated points — there is
no overlap length to clip. So the membrane remap is done not in physical space but
on a common 1D coordinate: arc length s ∈ [0, L) around the closed loop. Both
meshes are lifted onto s, their node sets merged into a *supermesh* (each
sub-segment lies inside exactly one old and one new segment), and the conservative
L2 projection ρ_new = M⁻¹ B ρ_old is assembled over the supermesh.

Conservation is structural: with the new P1 basis a partition of unity, the column
sums of the mixed mass matrix B equal the new mass matrix's row sums, so
1ᵀ M ρ_new = 1ᵀ B ρ_old = ∫_Γ ρ_old ds exactly.

This module is pure NumPy and operates entirely in arc-length space. Turning a
DOLFINx boundary mesh into (node coordinates, ordered loop) and writing ρ_new back
into a `fem.Function` — and the closest-point projection needed when the new mesh
does not lie on the old curve (non-co-located remesh) — are separate, later
increments. The kernel here is agnostic to where the arc-length coordinates came
from: co-located node positions give an exact remap, projected positions an
approximate one, but the conservation guarantee is identical either way.
"""

from __future__ import annotations

import itertools
from typing import cast

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]


def arclength_parameterization(points: Floats, *, closed: bool = True) -> tuple[Floats, float]:
    """Cumulative arc-length coordinate of an ordered polyline's nodes.

    `points` is an (N, d) array of node coordinates in traversal order. Returns
    `(s, length)` where `s[k]` is the arc length from `points[0]` to `points[k]`
    and `length` is the total curve length. For a closed loop (`closed=True`) the
    total length includes the closing segment from the last node back to the
    first, so every node satisfies `0 == s[0] < … < s[N-1] < length` — exactly the
    convention `supermesh_remap_1d` expects.
    """

    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("points must be an (N, d) array with N >= 2 nodes")

    # N points -> N-1 open-segment lengths -> N cumulative coords (node 0 at s=0).
    segment_lengths: Floats = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s: Floats = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if closed:
        closing = float(np.linalg.norm(points[0] - points[-1]))
        return s, float(s[-1]) + closing  # all N nodes lie in [0, length)
    return s, float(s[-1])


def project_points_to_polyline_arclength(points: Floats, loop_coords: Floats, s_loop: Floats, length: float) -> Floats:
    """Arc-length position on a closed polyline of each point's closest point.

    `loop_coords` is the (M, d) ordered node array of the reference loop and
    `s_loop` their arc-length coordinates (from `arclength_parameterization`);
    `length` is the loop's total length. For each row of `points` (d-dimensional),
    the nearest point on the polyline — over all segments including the wrap-around
    closing segment — is found and its arc-length position returned, in `[0, length)`.

    This is the step that lifts a *new* membrane mesh onto the *old* mesh's
    arc-length frame so `supermesh_remap_1d` can remap between them. When the new
    nodes lie exactly on the old polyline (a co-located remesh) the projection is
    exact; otherwise it is the closest-point approximation the design note flags.
    """

    if points.ndim != 2 or loop_coords.ndim != 2 or points.shape[1] != loop_coords.shape[1]:
        raise ValueError("points and loop_coords must be (N, d) and (M, d) with matching d")

    starts = loop_coords
    edges = np.roll(loop_coords, -1, axis=0) - loop_coords  # segment vectors, last = closing
    edge_len = np.linalg.norm(edges, axis=1)
    denom = np.sum(edges * edges, axis=1)

    out = np.empty(points.shape[0])
    for k, p in enumerate(points):
        offset = p - starts
        t = np.clip(np.sum(offset * edges, axis=1) / denom, 0.0, 1.0)
        closest = starts + t[:, None] * edges
        seg = int(np.argmin(np.sum((p - closest) ** 2, axis=1)))
        out[k] = (s_loop[seg] + t[seg] * edge_len[seg]) % length
    return out


def supermesh_remap_1d(s_old: Floats, rho_old: Floats, s_new: Floats, length: float) -> Floats:
    """Conservatively remap a P1 surface density between two loop discretizations.

    Both `s_old` and `s_new` are strictly increasing arc-length coordinates in
    `[0, length)` of the nodes of a *closed* membrane (segments connect consecutive
    nodes, with a wrap-around segment from the last node back to the first).
    `rho_old` holds the nodal ρ values at `s_old`. Returns `rho_new` at `s_new`
    such that ∫_Γ ρ_new ds == ∫_Γ ρ_old ds to round-off.
    """

    _validate_loop(s_old, length, "s_old")
    _validate_loop(s_new, length, "s_new")
    if rho_old.shape != s_old.shape:
        raise ValueError("rho_old must have one value per s_old node")

    n_new = s_new.size
    n_old = s_old.size
    mass_new = np.zeros((n_new, n_new))  # ∫ φ_new_j φ_new_k ds
    mixed = np.zeros((n_new, n_old))  # ∫ φ_new_j φ_old_i ds

    # Supermesh breakpoints: every node of either mesh. The loop is traversed once
    # by appending the first break shifted by one period, so the final sub-segment
    # is the wrap-around piece [breaks[-1], breaks[0] + length].
    breaks = np.unique(np.concatenate((s_old, s_new)))
    breaks_ext: Floats = np.concatenate((breaks, [breaks[0] + length]))

    for a, b in itertools.pairwise(breaks_ext):
        h = float(b - a)
        if h <= 0.0:
            continue
        mid = 0.5 * (a + b)
        o_left, o_right, o_a, o_b = _basis_on_segment(s_old, length, mid, a, b)
        n_left, n_right, n_a, n_b = _basis_on_segment(s_new, length, mid, a, b)

        # The two active hats of each parent segment, indexed (left, right), with
        # their values at the sub-segment ends a and b. The returned `*_a` / `*_b`
        # are the right-node weights; the left node carries 1 - that.
        new_nodes = (n_left, n_right)
        new_wa, new_wb = (1.0 - n_a, n_a), (1.0 - n_b, n_b)
        old_nodes = (o_left, o_right)
        old_wa, old_wb = (1.0 - o_a, o_a), (1.0 - o_b, o_b)

        for p in (0, 1):
            j = new_nodes[p]
            for q in (0, 1):
                mass_new[j, new_nodes[q]] += _l2_linear(new_wa[p], new_wb[p], new_wa[q], new_wb[q], h)
            for r in (0, 1):
                mixed[j, old_nodes[r]] += _l2_linear(new_wa[p], new_wb[p], old_wa[r], old_wb[r], h)

    return cast(Floats, np.linalg.solve(mass_new, mixed @ rho_old))


def _validate_loop(s: Floats, length: float, name: str) -> None:
    if s.ndim != 1 or s.size < 2:
        raise ValueError(f"{name} must be a 1D array of at least 2 nodes")
    if s[0] < 0.0 or s[-1] >= length:
        raise ValueError(f"{name} nodes must lie in [0, {length})")
    if not np.all(np.diff(s) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _basis_on_segment(s_nodes: Floats, length: float, mid: float, a: float, b: float) -> tuple[int, int, float, float]:
    """For the sub-segment [a, b] (parent identified by its midpoint `mid`), return
    the two active node indices of `s_nodes` and the right-node hat weight at `a`
    and at `b`. The sub-segment lies wholly inside one parent segment, so each hat
    is linear across it and two endpoint evaluations determine it.

    The parent is found from `mid` (not from `a`/`b`, which coincide with node
    positions and would otherwise resolve to adjacent segments). The wrap-around
    segment — from the last node up to the first node plus one period — is handled
    by extending its right endpoint past `length`.
    """

    n = s_nodes.size
    i = int(np.searchsorted(s_nodes, mid % length, side="right")) - 1
    if 0 <= i <= n - 2:
        left, right = i, i + 1
        s_left, s_right = float(s_nodes[i]), float(s_nodes[i + 1])
        wrap = False
    else:  # mid is in the wrap-around segment (before the first node or after the last)
        left, right = n - 1, 0
        s_left, s_right = float(s_nodes[-1]), float(s_nodes[0]) + length
        wrap = True
    span = s_right - s_left
    return left, right, _right_weight(a, s_left, span, wrap, length), _right_weight(b, s_left, span, wrap, length)


def _right_weight(x: float, s_left: float, span: float, wrap: bool, length: float) -> float:
    xe = x + length if (wrap and x < s_left) else x
    return (xe - s_left) / span


def _l2_linear(fa: float, fb: float, ga: float, gb: float, h: float) -> float:
    """∫₀ʰ f·g for f, g linear with endpoint values (fa, fb), (ga, gb)."""

    return h * (2.0 * fa * ga + fa * gb + fb * ga + 2.0 * fb * gb) / 6.0
