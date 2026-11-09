"""The outward unit normal of a moving membrane, as a nodal field on a bulk mesh (ADR 013 decision 2).

A front velocity that reads `geom.normal` is evaluated where the mesh moves: at the nodes of the
bulk mesh's boundary that *is* the membrane. The normal there is the area-weighted facet normal
`∫_Γ n·φ_i ds`, normalized at each boundary node. Interior nodes get 0 (no NaN), which is harmless:
a bulk mesh's motion reads only its boundary values (`_HarmonicExtension`).

The orientation is §1.8.4's: on a closed membrane, pointing away from the region it encloses. On an
enclosed volume (the single-mesh case) that is the volume's own outward facet normal. On a bulk
*outside* the membrane (the coupled case, an annulus around a cell) it is the opposite. The side is
not assumed: for the normal pointing out of the enclosed region E, the divergence theorem gives
`∫_Γ x·n ds = d·|E| > 0` on a closed membrane, whatever its shape, so the sign of that integral
orients the field.

The field is refreshed by `refresh()` on the current configuration (after each mesh move, before the
velocity is interpolated), so the normal is the current one (§1.8.4, §1.10.6).
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, la
from dolfinx import mesh as dmesh
from mpi4py import MPI
from numpy.typing import NDArray


class BoundaryNormal:
    """A P1 vector field on `mesh` holding the membrane's outward unit normal at the nodes of the
    `facets` (all exterior facets when omitted), 0 elsewhere."""

    def __init__(self, mesh: dmesh.Mesh, facets: NDArray[np.int32] | None = None) -> None:
        gdim = mesh.geometry.dim
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        if facets is None:
            facets = dmesh.exterior_facet_indices(mesh.topology)
        facets = np.unique(np.asarray(facets, dtype=np.int32))
        tags = dmesh.meshtags(mesh, fdim, facets, np.ones(facets.size, dtype=np.int32))
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)(1)
        self._space = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
        self.function = fem.Function(self._space, name="normal")
        n = ufl.FacetNormal(mesh)
        self._weighted = fem.form(ufl.inner(n, ufl.TestFunction(self._space)) * ds)
        self._x_dot_n = fem.form(ufl.inner(ufl.SpatialCoordinate(mesh), n) * ds)
        self._gdim = gdim
        self._comm = mesh.comm
        self.refresh()

    def refresh(self) -> None:
        """Recompute the normal on the mesh's current configuration."""

        vector = fem.assemble_vector(self._weighted)
        vector.scatter_reverse(la.InsertMode.add)
        values = vector.array.reshape((-1, self._gdim)).copy()
        norms = np.linalg.norm(values, axis=1)
        on_boundary = norms > 0.0
        values[on_boundary] /= norms[on_boundary, None]
        values[~on_boundary] = 0.0
        self.function.x.array[: values.size] = values.ravel() * self._orientation()
        self.function.x.scatter_forward()

    def _orientation(self) -> float:
        moment = self._comm.allreduce(float(np.real(fem.assemble_scalar(self._x_dot_n))), op=MPI.SUM)
        return 1.0 if moment > 0.0 else -1.0
