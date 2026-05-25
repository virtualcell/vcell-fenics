"""2D circular cell + boundary submesh (Approach B)."""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI


@dataclass
class DiskMembrane:
    bulk_mesh: dmesh.Mesh
    submesh: dmesh.Mesh
    entity_map: dmesh.EntityMap
    boundary_facets: np.ndarray


def create_disk_with_membrane(
    radius: float = 1.0,
    h: float = 0.1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> DiskMembrane:
    """Create a 2D disk and extract its boundary as a codim-1 submesh.

    The bulk mesh is kept around so callers can later add bulk-surface
    coupling without re-meshing. For surface-only prototypes the bulk is
    unused.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("disk")
        gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
        gmsh.model.occ.synchronize()

        surf_tags = [s[1] for s in gmsh.model.getEntities(2)]
        curve_tags = [c[1] for c in gmsh.model.getEntities(1)]
        gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="bulk")
        gmsh.model.addPhysicalGroup(1, curve_tags, tag=2, name="membrane")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)

        data = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    bulk_mesh = data.mesh
    facet_tags = data.facet_tags
    assert facet_tags is not None, "model_to_mesh returned no facet tags for the 'membrane' physical group"
    boundary_facets = facet_tags.find(2)

    tdim = bulk_mesh.topology.dim
    submesh, entity_map, _vertex_map, _node_map = dmesh.create_submesh(bulk_mesh, tdim - 1, boundary_facets)

    return DiskMembrane(
        bulk_mesh=bulk_mesh,
        submesh=submesh,
        entity_map=entity_map,
        boundary_facets=np.asarray(boundary_facets),
    )


def scale_radially(submesh: dmesh.Mesh, factor: float) -> None:
    """Scale submesh node positions about the origin by ``factor``."""
    x = submesh.geometry.x
    x[:, 0] *= factor
    x[:, 1] *= factor
