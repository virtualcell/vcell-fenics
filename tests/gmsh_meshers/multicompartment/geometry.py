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

from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

from vcell_fenics.backend.geometry import (
    BoundaryGeometry,
    CoupledGeometry,
    Geometry,
    SubdomainGeometry,
)

CYTOSOL_TAG = 1
EXTRACELLULAR_TAG = 2
MEMBRANE_TAG = 3
OUTER_TAG = 4


@dataclass
class ExtracellularAnnulus:
    """The §1.6.6 substrate: a single extracellular bulk (an annulus) whose *inner*
    boundary is the membrane (Γ_mem) and whose *outer* boundary is the reservoir
    (∂Ω_outer). No cytoplasm is modelled. `membrane_mesh` is the codim-1 submesh of
    the inner circle, and `membrane_entity_map` relates it to `bulk_mesh` so a form
    over the bulk's membrane facets can reference membrane functions."""

    bulk_mesh: dmesh.Mesh
    facet_tags: dmesh.MeshTags
    membrane_mesh: dmesh.Mesh
    membrane_entity_map: dmesh.EntityMap
    inner_radius: float
    outer_radius: float


def create_extracellular_annulus(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    h: float = 0.1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> ExtracellularAnnulus:
    """An annular extracellular bulk: the outer disk with the inner (cell) disk cut
    out. The inner circle is tagged `MEMBRANE_TAG`, the outer `OUTER_TAG`; the
    membrane is extracted as a codim-1 submesh."""

    if not 0.0 < inner_radius < outer_radius:
        raise ValueError(f"need 0 < inner_radius < outer_radius, got {inner_radius} and {outer_radius}")

    import gmsh  # lazy: keep gmsh (GPL) out of the default import graph — loaded only when this prototype mesher runs

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("extracellular_annulus")
        outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, outer_radius, outer_radius)
        inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, inner_radius, inner_radius)
        gmsh.model.occ.cut([(2, outer)], [(2, inner)])  # the annulus
        gmsh.model.occ.synchronize()

        surfaces = [s[1] for s in gmsh.model.getEntities(2)]
        lengths = {c[1]: gmsh.model.occ.getMass(1, c[1]) for c in gmsh.model.getEntities(1)}
        membrane_curve = min(lengths, key=lambda t: lengths[t])  # inner circle
        outer_curve = max(lengths, key=lambda t: lengths[t])

        gmsh.model.addPhysicalGroup(2, surfaces, tag=EXTRACELLULAR_TAG, name="extracellular")
        gmsh.model.addPhysicalGroup(1, [membrane_curve], tag=MEMBRANE_TAG, name="membrane")
        gmsh.model.addPhysicalGroup(1, [outer_curve], tag=OUTER_TAG, name="outer")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        data = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    bulk = data.mesh
    facet_tags = data.facet_tags
    assert facet_tags is not None, "model_to_mesh returned no facet tags for the membrane / outer groups"
    membrane_mesh, entity_map, *_ = dmesh.create_submesh(bulk, bulk.topology.dim - 1, facet_tags.find(MEMBRANE_TAG))

    return ExtracellularAnnulus(
        bulk_mesh=bulk,
        facet_tags=facet_tags,
        membrane_mesh=membrane_mesh,
        membrane_entity_map=entity_map,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )


@dataclass
class CellExtracellular:
    """A concentric two-compartment cell. `parent_mesh` is the union mesh; `cell_tags`
    marks each cell's compartment (CYTOSOL_TAG / EXTRACELLULAR_TAG) and `facet_tags`
    marks the membrane and outer curves (MEMBRANE_TAG / OUTER_TAG). The three
    submeshes are the per-compartment bulk meshes and the codim-1 membrane.

    The three `*_entity_map`s relate each submesh's entities back to `parent_mesh`, so a
    cross-mesh form integrated on the parent's interface facets can pull in functions from
    all three submeshes (the two-sided-trace substrate — a membrane equation referencing
    `trace(u_cytosol)` and `trace(u_extracellular)` at once)."""

    parent_mesh: dmesh.Mesh
    cell_tags: dmesh.MeshTags
    facet_tags: dmesh.MeshTags
    cytosol_mesh: dmesh.Mesh
    extracellular_mesh: dmesh.Mesh
    membrane_mesh: dmesh.Mesh
    cytosol_entity_map: dmesh.EntityMap
    extracellular_entity_map: dmesh.EntityMap
    membrane_entity_map: dmesh.EntityMap
    inner_radius: float
    outer_radius: float
    # The two compartments' cell tags on `parent_mesh`, named by *topology* (inner disk / outer
    # annulus) rather than biology — so a math-layer consumer (`backend`) can use them without
    # importing the cytosol/extracellular naming. Here: CYTOSOL_TAG / EXTRACELLULAR_TAG.
    inner_region_tag: int = CYTOSOL_TAG
    outer_region_tag: int = EXTRACELLULAR_TAG


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

    import gmsh  # lazy: keep gmsh (GPL) out of the default import graph — loaded only when this mesher runs

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
    cytosol_mesh, cytosol_emap, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(CYTOSOL_TAG))
    extracellular_mesh, extracellular_emap, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(EXTRACELLULAR_TAG))
    membrane_mesh, membrane_emap, *_ = dmesh.create_submesh(parent, tdim - 1, facet_tags.find(MEMBRANE_TAG))

    return CellExtracellular(
        parent_mesh=parent,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        cytosol_mesh=cytosol_mesh,
        extracellular_mesh=extracellular_mesh,
        membrane_mesh=membrane_mesh,
        cytosol_entity_map=cytosol_emap,
        extracellular_entity_map=extracellular_emap,
        membrane_entity_map=membrane_emap,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )


# ---------------------------------------------------------------------------
# Backend-`Geometry` builders that wrap the gmsh meshers above.
#
# These used to live in `vcell_fenics.backend.geometry`, but the production geometry path is now the
# gmsh-free Netgen `realize` (see LICENSING.md). They are retained here — beside the gmsh meshers they
# wrap — as the annular-reservoir / concentric-annulus variants the coupled tests exercise, which the
# box-bounded `realize_interface_coupled` does not reproduce exactly (true circular outer edge, annulus
# extracellular bulk). Test-only: nothing in `src/` imports them.
# ---------------------------------------------------------------------------


def make_cell_extracellular_geometry(
    name: str,
    *,
    cytosol: str,
    extracellular: str,
    membrane: str,
    interface: str,
    outer: str,
    inner_radius: float = 0.6,
    outer_radius: float = 1.0,
    h: float = 0.1,
) -> Geometry:
    """A concentric two-compartment cell: an inner-disk `cytosol` and outer-annulus `extracellular`
    (both `volume`), plus the `membrane` between them as a `surface` subdomain. The membrane is also
    registered as the internal boundary `interface` (incident to both compartments); the outer circle is
    the external boundary `outer` (incident to the extracellular space only)."""

    cell = create_cell_extracellular(inner_radius=inner_radius, outer_radius=outer_radius, h=h)
    subdomains = {
        cytosol: SubdomainGeometry(mesh=cell.cytosol_mesh, kind="volume"),
        extracellular: SubdomainGeometry(mesh=cell.extracellular_mesh, kind="volume"),
        membrane: SubdomainGeometry(mesh=cell.membrane_mesh, kind="surface"),
    }
    boundaries = {
        interface: BoundaryGeometry(subdomains=(cytosol, extracellular), facets=cell.facet_tags.find(MEMBRANE_TAG)),
        outer: BoundaryGeometry(subdomains=(extracellular,), facets=cell.facet_tags.find(OUTER_TAG)),
    }
    return Geometry(
        name=name,
        subdomains=subdomains,
        boundaries=boundaries,
        parent_mesh=cell.parent_mesh,
        cell_tags=cell.cell_tags,
        facet_tags=cell.facet_tags,
    )


def make_extracellular_annulus_geometry(
    name: str,
    *,
    extracellular: str,
    membrane: str,
    interface: str,
    outer: str,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    h: float = 0.1,
) -> CoupledGeometry:
    """The §1.6.6 coupled geometry: an annular `extracellular` bulk whose inner boundary is the
    `membrane` surface subdomain (coupled at `interface`) and whose outer boundary is `outer` (the
    reservoir)."""

    annulus = create_extracellular_annulus(inner_radius=inner_radius, outer_radius=outer_radius, h=h)
    return CoupledGeometry(
        name=name,
        bulk_subdomain=extracellular,
        surface_subdomain=membrane,
        bulk_mesh=annulus.bulk_mesh,
        surface_mesh=annulus.membrane_mesh,
        entity_map=annulus.membrane_entity_map,
        facet_tags=annulus.facet_tags,
        interface=interface,
        interface_tag=MEMBRANE_TAG,
        outer=outer,
        outer_tag=OUTER_TAG,
    )
