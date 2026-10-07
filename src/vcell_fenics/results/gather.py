"""A canonical, MPI-rank-count-independent point and cell order for a domain's P1 output.

Every P1 dof sits on a mesh geometry node, and DOLFINx carries each node's index in the original
(rank-0) mesh input as ``mesh.geometry.input_global_indices`` — through ``create_submesh`` too. Keying
owned dofs by that index and sorting on rank 0 gives the same point order at any rank count
(ADR 010 §6(f)); cells are canonicalized (sorted vertices, positively oriented when full-dimensional,
lexicographically ordered) so the VTU a domain writes is byte-identical serially and under MPI.

Collective: construct :class:`P1Layout` and call :meth:`P1Layout.gather` on every rank; only rank 0
receives arrays (the others get ``None``).
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from numpy.typing import NDArray

from vcell_fenics.results.vtu import VTK_LINE, VTK_TETRA, VTK_TRIANGLE

_VTK_SIMPLEX = {1: VTK_LINE, 2: VTK_TRIANGLE, 3: VTK_TETRA}


class P1Layout:
    """The canonical layout of a scalar degree-1 Lagrange space on a simplex mesh."""

    def __init__(self, space: fem.FunctionSpace) -> None:
        mesh = space.mesh
        self._mesh = mesh
        self._comm = mesh.comm
        tdim = mesh.topology.dim
        if tdim not in _VTK_SIMPLEX or mesh.topology.cell_name() not in ("interval", "triangle", "tetrahedron"):
            raise ValueError(f"bundle output supports simplex meshes only, not {mesh.topology.cell_name()!r}")
        if space.dofmap.index_map_bs != 1 or space.element.basix_element.degree != 1:
            raise ValueError("P1Layout needs a scalar degree-1 Lagrange space")
        self.tdim = tdim
        self.gdim = mesh.geometry.dim
        self.vtk_type = _VTK_SIMPLEX[tdim]

        dof_map = space.dofmap.index_map
        self._n_owned = int(dof_map.size_local)
        dofs = np.asarray(space.dofmap.list)
        nodes = np.asarray(mesh.geometry.dofmap)
        input_index = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        node_of_dof = np.full(self._n_owned + dof_map.num_ghosts, -1, dtype=np.int64)
        node_of_dof[dofs.ravel()] = nodes.ravel()  # P1 dofs and P1 geometry nodes share a cell layout
        if (node_of_dof < 0).any():
            raise RuntimeError("a P1 dof is not attached to any cell's geometry node")
        key = input_index[node_of_dof]
        self._owned_nodes = node_of_dof[: self._n_owned]

        # The node's own coordinates, not `tabulate_dof_coordinates` — that pushes a reference point
        # through whichever cell it meets last, which differs with the partition by an ulp.
        coords = np.asarray(mesh.geometry.x)[node_of_dof[: self._n_owned], : self.gdim]
        n_cells_owned = mesh.topology.index_map(tdim).size_local
        gathered_keys = self._comm.gather(key[: self._n_owned], root=0)
        gathered_coords = self._comm.gather(coords, root=0)
        gathered_cells = self._comm.gather(key[dofs[:n_cells_owned]], root=0)

        self._order: NDArray[np.int64] | None = None
        self._keys: NDArray[np.int64] | None = None
        self._points: NDArray[np.float64] | None = None
        self._cells: NDArray[np.int64] | None = None
        n_points = n_cells = 0
        if self._comm.rank == 0:
            assert gathered_keys is not None and gathered_coords is not None and gathered_cells is not None
            keys = np.concatenate(gathered_keys)
            self._order = np.argsort(keys, kind="stable")
            sorted_keys = keys[self._order]
            if sorted_keys.size > 1 and not np.all(np.diff(sorted_keys) > 0):
                raise RuntimeError("P1 point keys are not unique across ranks")
            self._keys = sorted_keys
            self._points = np.concatenate(gathered_coords)[self._order]
            cell_keys = np.concatenate(gathered_cells)
            positions = np.searchsorted(sorted_keys, cell_keys)
            self._cells = _canonical_cells(positions.astype(np.int64), self._points, full=tdim == self.gdim)
            n_points, n_cells = int(self._points.shape[0]), int(self._cells.shape[0])
        self.n_points, self.n_cells = self._comm.bcast((n_points, n_cells), root=0)

    @property
    def n_owned(self) -> int:
        """Owned dofs on this rank — the length :meth:`gather` expects."""

        return self._n_owned

    def points(self) -> NDArray[np.float64] | None:
        """(n_points, gdim) coordinates in canonical order — rank 0 only."""

        return self._points

    def keys(self) -> NDArray[np.int64] | None:
        """(n_points,) each point's mesh-input index, in canonical (increasing) order — rank 0 only. Two
        submeshes of one parent mesh share the parent's input indices, so equal keys are the same vertex."""

        return self._keys

    def cells(self) -> NDArray[np.int64] | None:
        """(n_cells, tdim + 1) canonical point indices — rank 0 only."""

        return self._cells

    def gather_coords(self) -> NDArray[np.float64] | None:
        """The mesh's *current* point coordinates, (n_points, 3), in canonical order on rank 0 — for a
        moving (ALE) domain, whose geometry moves in place under the same topology and point order."""

        coords = np.ascontiguousarray(np.asarray(self._mesh.geometry.x)[self._owned_nodes, :], dtype=np.float64)
        parts = self._comm.gather(coords, root=0)
        if parts is None:
            return None
        assert self._order is not None
        gathered: NDArray[np.float64] = np.concatenate(parts)[self._order]
        return gathered

    def gather(self, owned: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Collect this rank's owned dof values into canonical order on rank 0."""

        if owned.shape[0] != self._n_owned:
            raise ValueError(f"expected {self._n_owned} owned values, got {owned.shape[0]}")
        parts = self._comm.gather(np.ascontiguousarray(owned, dtype=np.float64), root=0)
        if parts is None:
            return None
        assert self._order is not None
        values: NDArray[np.float64] = np.concatenate(parts)[self._order]
        return values


def _canonical_cells(cells: NDArray[np.int64], points: NDArray[np.float64], *, full: bool) -> NDArray[np.int64]:
    """Sort each cell's vertices, then — for a full-dimensional cell (a tetrahedron in 3D, a triangle in
    2D) — swap the last two if that leaves it negatively oriented; finally order the cells
    lexicographically. Deterministic for a given mesh, whatever order the cells were gathered in."""

    canonical = np.sort(cells, axis=1)
    if full and canonical.shape[1] >= 3:
        origin = points[canonical[:, 0]]
        edges = np.stack([points[canonical[:, k]] - origin for k in range(1, canonical.shape[1])], axis=1)
        negative = np.linalg.det(edges) < 0.0
        canonical[negative, -2:] = canonical[negative, -2:][:, ::-1]
    order = np.lexsort(np.sort(canonical, axis=1).T[::-1])
    return canonical[order]
