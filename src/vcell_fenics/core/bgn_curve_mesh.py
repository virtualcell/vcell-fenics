"""DOLFINx bridge for BGN tangential mesh redistribution.

`bgn_curve.py` is a pure-NumPy kernel that advances a closed *polyline* by one
semi-implicit BGN mean-curvature-flow step. This module connects it to a live
DOLFINx membrane `Mesh`: it orders the membrane's nodes into a loop (reusing
`ordered_membrane_loop`), runs the kernel, and writes the new node positions back
into `mesh.geometry.x` in place — so the membrane both advances by curvature flow
and keeps its nodes well-distributed.

The write-back has the same subtlety `_MeshMotion` handles: a P1 space's dof order
does not match the mesh's geometry-node order, so the per-dof displacement is mapped
onto geometry rows by a one-time closest-point match (`cKDTree`). The displacement
(not the absolute position) is what is scattered, exactly as the prescribed-motion
path does, so the two stay consistent.

Scope (v1): serial, a single closed-loop P1 membrane in 2D. Mean-curvature
(surface-tension) flow specifically — the mobility `m = σ/η` is the caller's (the
unknown-motion path calibrates it from the force balance). The companion ALE
advection term for a co-moving surface species (mesh velocity ≠ material velocity)
is the consumer's concern, not this geometric step's.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx.mesh import Mesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from vcell_fenics.core.bgn_curve import bgn_curvature_flow_step
from vcell_fenics.core.surface_remap_mesh import ordered_membrane_loop

Floats = NDArray[np.float64]


def bgn_redistribute_membrane(mesh: Mesh, *, mobility: float, dt: float) -> None:
    """Advance a closed P1 membrane `Mesh` by one BGN curvature-flow step, in place.

    Orders the membrane loop, runs `bgn_curvature_flow_step` (normal advance +
    tangential redistribution), and updates `mesh.geometry.x`. `mobility` is
    `m = σ/η` from the surface-tension force balance (normal velocity `V = −m κ`).
    Raises `ValueError` if the step collapses or inverts an edge.
    """

    if mesh.topology.dim != 1:
        raise ValueError(f"expected a codim-1 membrane (1D interval mesh), got topology dim {mesh.topology.dim}")
    gdim = mesh.geometry.dim

    scalar_space = fem.functionspace(mesh, ("Lagrange", 1))
    coords_all: Floats = scalar_space.tabulate_dof_coordinates()
    loop_coords, order = ordered_membrane_loop(scalar_space)  # (N, gdim), dof permutation

    new_coords = bgn_curvature_flow_step(loop_coords, mobility=mobility, dt=dt)
    if not _edges_positive(new_coords):
        raise ValueError("BGN step collapsed or inverted a membrane edge; reduce dt or the mobility")

    # Scatter the per-dof displacement onto geometry rows (dof order ≠ geometry order).
    n = coords_all.shape[0]
    disp_by_dof = np.zeros((n, gdim))
    disp_by_dof[order] = new_coords - loop_coords
    geom_from_dof = cKDTree(coords_all[:, :gdim]).query(mesh.geometry.x[:, :gdim])[1]
    mesh.geometry.x[:, :gdim] += disp_by_dof[geom_from_dof]


def _edges_positive(loop: Floats) -> bool:
    """Whether every edge of the closed loop has strictly positive length."""

    lengths = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
    return bool(lengths.min() > 1e-12)
