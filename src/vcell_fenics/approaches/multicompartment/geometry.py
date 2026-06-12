"""A concentric two-compartment cell geometry with an internal membrane interface.

The single-compartment geometries (`approaches/static`, `approaches/submesh`) give
one bulk region whose whole boundary is external. Multi-compartment models — two
bulk subdomains coupled across a shared membrane (interface BCs §1.6.2, bulk↔surface
trace coupling §1.6.6, the ALE bulk-trace case) — need an *internal* boundary: a
codim-1 entity incident to two subdomains, where the membrane separates them.

This builds the canonical structure: an inner disk Ω_cyto (the cytosol) nested in an
outer annulus Ω_ext (the extracellular space), meeting at the membrane Γ_mem, with
the outer circle ∂Ω_outer the only external boundary. gmsh fragments two concentric
disks into the two surfaces (sharing the inner circle), tagged by area / length; the
DOLFINx reader returns the parent mesh with cell tags (which compartment) and facet
tags (membrane vs outer). Each compartment and the membrane are then extracted as
submeshes — usable standalone, and the substrate for mixed-dimensional coupling.

The key property, verified in the tests: Γ_mem is *internal* (it integrates under
the interior-facet measure `dS`, not `ds`) and is incident to **both** compartments,
whereas ∂Ω_outer is external (incident to the extracellular space only).
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

CYTOSOL_TAG = 1
EXTRACELLULAR_TAG = 2
MEMBRANE_TAG = 3
OUTER_TAG = 4


@dataclass
class CellExtracellular:
    """A concentric two-compartment cell. `parent_mesh` is the union mesh; `cell_tags`
    marks each cell's compartment (CYTOSOL_TAG / EXTRACELLULAR_TAG) and `facet_tags`
    marks the membrane and outer curves (MEMBRANE_TAG / OUTER_TAG). The three
    submeshes are the per-compartment bulk meshes and the codim-1 membrane."""

    parent_mesh: dmesh.Mesh
    cell_tags: dmesh.MeshTags
    facet_tags: dmesh.MeshTags
    cytosol_mesh: dmesh.Mesh
    extracellular_mesh: dmesh.Mesh
    membrane_mesh: dmesh.Mesh
    inner_radius: float
    outer_radius: float


def create_cell_extracellular(
    inner_radius: float = 0.6,
    outer_radius: float = 1.0,
    h: float = 0.1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> CellExtracellular:
    """An inner disk (cytosol) inside an outer annulus (extracellular), meeting at the
    membrane. The two compartments share the inner circle as an internal interface;
    the outer circle is the only external boundary."""

    if not 0.0 < inner_radius < outer_radius:
        raise ValueError(f"need 0 < inner_radius < outer_radius, got {inner_radius} and {outer_radius}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cell_extracellular")
        outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, outer_radius, outer_radius)
        inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, inner_radius, inner_radius)
        # Fragment splits the outer disk into the inner disk + the annulus, sharing
        # the inner circle (so the membrane becomes one internal curve, not two).
        gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()

        # Identify the two surfaces by area (inner disk < annulus) and the two curves
        # by length (inner circle < outer circle) — robust to gmsh's entity numbering.
        areas = {s[1]: gmsh.model.occ.getMass(2, s[1]) for s in gmsh.model.getEntities(2)}
        lengths = {c[1]: gmsh.model.occ.getMass(1, c[1]) for c in gmsh.model.getEntities(1)}
        cytosol_surf = min(areas, key=lambda t: areas[t])
        extracellular_surf = max(areas, key=lambda t: areas[t])
        membrane_curve = min(lengths, key=lambda t: lengths[t])
        outer_curve = max(lengths, key=lambda t: lengths[t])

        gmsh.model.addPhysicalGroup(2, [cytosol_surf], tag=CYTOSOL_TAG, name="cytosol")
        gmsh.model.addPhysicalGroup(2, [extracellular_surf], tag=EXTRACELLULAR_TAG, name="extracellular")
        gmsh.model.addPhysicalGroup(1, [membrane_curve], tag=MEMBRANE_TAG, name="membrane")
        gmsh.model.addPhysicalGroup(1, [outer_curve], tag=OUTER_TAG, name="outer")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        data = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    parent = data.mesh
    cell_tags = data.cell_tags
    facet_tags = data.facet_tags
    assert cell_tags is not None, "model_to_mesh returned no cell tags for the compartment physical groups"
    assert facet_tags is not None, "model_to_mesh returned no facet tags for the membrane / outer groups"

    tdim = parent.topology.dim
    cytosol_mesh, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(CYTOSOL_TAG))
    extracellular_mesh, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(EXTRACELLULAR_TAG))
    membrane_mesh, *_ = dmesh.create_submesh(parent, tdim - 1, facet_tags.find(MEMBRANE_TAG))

    return CellExtracellular(
        parent_mesh=parent,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        cytosol_mesh=cytosol_mesh,
        extracellular_mesh=extracellular_mesh,
        membrane_mesh=membrane_mesh,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
