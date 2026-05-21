"""Fixed-geometry meshes for the static approach.

Bulk-only here; surface-on-boundary-trace and bulk-surface coupling come
later. We keep this small until we have multiple consumers to factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

BOUNDARY_TAG = 2


@dataclass
class StaticDisk:
    mesh: dmesh.Mesh
    facet_tags: dmesh.MeshTags
    radius: float


def create_disk(
    radius: float = 1.0,
    h: float = 0.1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> StaticDisk:
    """2D disk with its outer boundary tagged ``BOUNDARY_TAG``."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("disk")
        gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
        gmsh.model.occ.synchronize()

        surf_tags = [s[1] for s in gmsh.model.getEntities(2)]
        curve_tags = [c[1] for c in gmsh.model.getEntities(1)]
        gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="bulk")
        gmsh.model.addPhysicalGroup(1, curve_tags, tag=BOUNDARY_TAG, name="boundary")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        data = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    return StaticDisk(mesh=data.mesh, facet_tags=data.facet_tags, radius=radius)
