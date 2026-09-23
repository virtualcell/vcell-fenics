"""Realize a :class:`~vcell_fenics.formalism.geometry_schema.GeometryDescription` into a concrete
backend :class:`~vcell_fenics.backend.geometry.Geometry` (ADR 007; `geometric-formalism.md` §3).

This is the bridge from the declarative geometry *spec* to a meshed object the backend solves on —
the realization layer §3.1 describes. It subsumes the imperative ``make_*`` helpers as recipes over
the formalism.

**Realization v1:**

- **Trivial / non-spatial (``dim = 0``).** A well-mixed geometry: each ``compartmental`` subvolume
  (and any membrane between them) is backed by a minimal single-cell mesh the lumped-ODE templates
  carry a constant over — a representational stand-in, not a spatial domain.
- **2D single-subvolume whole box (``dim = 2``).** A geometry with one subvolume (VCell's analytic
  background ``'1.0'``) is the bounding box itself: a plain structured mesh, one ``volume`` region,
  the four box faces named — no contours or membranes (the single-compartment pattern).
- **2D body-fitted (``dim = 2``), box-partitioned, multi-region.** The geometry is the bounding box
  (``extent`` / ``origin``) partitioned by ``N`` analytic subvolumes (the last is the background /
  complement); the outer boundary is the **box faces** (``x_minus`` / ``x_plus`` / ``y_minus`` /
  ``y_plus``) and each ``SurfaceClass`` is an internal membrane. Each non-background subvolume's
  boolean predicate is lowered to a Rvachev implicit field (`formalism/rvachev.py`), sampled on a
  grid, and its ``φ = 0`` boundary is **marched** (scikit-image) and embedded as a conforming internal
  boundary in a Netgen model that partitions the box into the regions (nested shapes nest; disjoint
  shapes sit side by side; LGPL Netgen replaces GPL gmsh — ADR 008). Each cell is then assigned to the
  subvolume whose priority-resolved field is negative at its midpoint; membrane facets between two
  regions are named by the ``SurfaceClass`` for that pair.

- **3D body-fitted (``dim = 3``)**: the same recipe with marching cubes and a Netgen volume mesh.
- **Image geometries (2D and 3D)**, any topology — nested regions, regions cut by the box, and
  junctions where three or more subvolumes meet: the image's smoothed label field (`labels.py`), the
  conforming boundaries between its subvolumes (`label_surfaces.py`, VTK SurfaceNets with a sentinel
  label per box face), and a Netgen mesh with those boundaries embedded — Netgen's domain is the
  region, so no classification is needed (:func:`_realize_image_partition`).

**Not yet here:** analytic subvolumes touching the box boundary, and the unfitted (level-set /
cut-FEM) consumption of the same field. Unsupported descriptions raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import numpy as np
import pyngcore
import ufl
from dolfinx import mesh as dmesh
from mpi4py import MPI
from numpy.typing import NDArray
from skimage.measure import approximate_polygon, find_contours, marching_cubes

from vcell_fenics.backend.geometry import (
    BoundaryGeometry,
    Geometry,
    InterfaceCoupledGeometry,
    SubdomainGeometry,
)
from vcell_fenics.backend.implicit_fields import RealizationError as RealizationError  # re-exported
from vcell_fenics.backend.implicit_fields import eval_field as _eval_field
from vcell_fenics.backend.label_surfaces import LabelBoundary, extract_boundary
from vcell_fenics.backend.labels import label_geometry
from vcell_fenics.formalism.expr import Expr
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import lower_predicate, subvolume_implicit_functions

# Netgen's default multi-threaded TaskManager busy-waits in a long-lived process (ADR 008 §3); the
# realization meshes are small and serial, so cap the pool before the mesher loads. Its import must
# follow the cap, so it sits after the other imports (E402) rather than in the sorted block above.
pyngcore.SetNumThreads(1)

from netgen.csg import CSGeometry, OrthoBrick  # noqa: E402
from netgen.csg import Pnt as CsgPnt  # noqa: E402
from netgen.geom2d import SplineGeometry  # noqa: E402  (must follow SetNumThreads)
from netgen.meshing import Element2D, FaceDescriptor  # noqa: E402
from netgen.meshing import Mesh as NetgenMesh  # noqa: E402
from netgen.stl import STLGeometry  # noqa: E402

_FACE_NAMES_2D = ("x_minus", "x_plus", "y_minus", "y_plus")
_FACE_NAMES_3D = ("x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus")
# Mesh-tag layout for the 2D realization: one tag per declared surface class (membrane), one per box
# face. Region (cell) tags are 1..N over the subvolumes.
_SURFACE_TAG_BASE = 100  # membrane tags, in SurfaceClass order
_FACE_TAG_BASE = 200  # x_minus=200, x_plus=201, …
_OUTER_WALL_TAG = 300  # the whole box exterior, unioned, as one reservoir "wall" (realize_interface_coupled)

# An image geometry is meshed serially on rank 0; refuse a mesh size that would make that intractable.
# Tetrahedra per unit volume at edge length h ≈ 6√2 / h³ (a regular tetrahedron has volume h³ / (6√2)).
_MAX_IMAGE_TETS = 4_000_000


class ImageGeometryWarning(UserWarning):
    """Realizing an image geometry changed its topology at this mesh size, or found a touching pair of
    subvolumes with no surface class (see :func:`~vcell_fenics.backend.labels.label_geometry`)."""


@dataclass(frozen=True)
class _Tagging:
    """The result of classifying/tagging a realized mesh: the cell and facet `MeshTags`, the
    subvolume-name → region-tag and surface-name → membrane-tag maps, and which subvolume(s) each
    box face is incident to."""

    cell_tags: dmesh.MeshTags
    facet_tags: dmesh.MeshTags
    region_tags: dict[str, int]
    surface_tags: dict[str, int]
    face_regions: dict[int, tuple[str, ...]]


def realize(
    description: GeometryDescription,
    *,
    h: float = 0.05,
    resolution: int | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Realize ``description`` into a backend :class:`Geometry`.

    ``h`` is the target mesh size. ``resolution`` is the per-axis sampling grid for the implicit field
    (2D only); leave it ``None`` to **resample at the mesh scale** — the marching grid spacing is set
    to ≈ ``h`` (``resolution ≈ extent / h``), so refining the mesh refines the faceted geometry with it
    and the realization converges to the exact analytic shape (a fixed resolution would leave a
    geometry-error floor, or over-resolve a coarse mesh into a degraded membrane). The result is *not*
    registered — the caller passes it to :func:`~vcell_fenics.backend.geometry.register_geometry` if
    name resolution is wanted.
    """

    if description.dim == 0:
        return _realize_trivial(description, comm)
    if description.dim == 2:
        return _realize_2d(description, h=h, resolution=resolution, comm=comm)
    if description.dim == 3:
        return _realize_3d(description, h=h, resolution=resolution, comm=comm)
    raise NotImplementedError(
        f"realization of dim={description.dim} geometry {description.name!r} is not implemented yet "
        "(supports dim 0, 2, 3)"
    )


# -- dim 0: trivial / well-mixed ------------------------------------------------


def _realize_trivial(description: GeometryDescription, comm: MPI.Comm) -> Geometry:
    if not description.subvolumes:
        raise RealizationError(f"geometry {description.name!r} has no subvolumes to realize")

    subdomains: dict[str, SubdomainGeometry] = {}
    for subvolume in description.subvolumes:
        if subvolume.type != "compartmental":
            raise RealizationError(
                f"non-spatial (dim 0) geometry {description.name!r} has subvolume {subvolume.name!r} "
                f"of type {subvolume.type!r}; a well-mixed geometry's subvolumes must be 'compartmental'"
            )
        subdomains[subvolume.name] = SubdomainGeometry(mesh=_lumped_mesh(comm), kind="volume")
    for surface in description.surfaces:
        subdomains[surface.name] = SubdomainGeometry(mesh=_lumped_mesh(comm), kind="surface")

    return Geometry(name=description.name, subdomains=subdomains)


def _lumped_mesh(comm: MPI.Comm) -> dmesh.Mesh:
    """A minimal single-cell mesh standing in for a non-spatial, well-mixed compartment."""

    return dmesh.create_unit_interval(comm, 1)


# -- dim 2: body-fitted, box-partitioned ----------------------------------------


def _realize_2d_partition(
    description: GeometryDescription, *, h: float, resolution: int | None, comm: MPI.Comm
) -> tuple[dmesh.Mesh, _Tagging]:
    """Body-fit and tag a multi-subvolume 2D geometry — the shared core of `_realize_2d` (which builds
    a plain `Geometry`) and `realize_interface_coupled` (which builds an `InterfaceCoupledGeometry`,
    retaining the submesh entity maps). Returns the parent mesh and its region / surface tagging."""

    if _is_image(description):
        return _realize_image_partition(description, h=h, comm=comm)
    subvolumes = description.subvolumes
    # Every subvolume but the last (the background / complement) defines an analytic shape we
    # body-fit to; the background owns whatever is left.
    for subvolume in subvolumes[:-1]:
        if subvolume.type != "analytic" or subvolume.expression is None:
            raise RealizationError(
                f"2D geometry {description.name!r} subvolume {subvolume.name!r} must be 'analytic' with an "
                f"expression to be realized (got type {subvolume.type!r})"
            )

    ox, oy = description.origin[0], description.origin[1]
    lx, ly = description.extent[0], description.extent[1]
    if resolution is None:
        # Resample at the mesh scale, **per axis** (≈ h spacing on EACH axis). A single point count for
        # both axes would under-sample the long axis of a non-square box (e.g. a unit disk in a 12×4 box:
        # dx = 0.08 but dy = 0.027). Per-axis counts keep dx ≈ dy ≈ h. (Clamped to keep a coarse mesh from
        # under-sampling and a fine one from blowing up.)
        nx, ny = (min(2001, max(51, round(length / h) + 1)) for length in (lx, ly))
    else:
        nx = ny = resolution

    # March each shape's own (raw, not priority-resolved) boundary so the mesh conforms to every
    # analytic surface; priority then decides each cell's owner. For nested shapes (nucleus in
    # cytosol in ecm) the contours nest; for disjoint shapes they sit side by side.
    contours: list[NDArray[np.float64]] = []
    for subvolume in subvolumes[:-1]:
        assert subvolume.expression is not None  # checked above
        raw_field = lower_predicate(parse(subvolume.expression))
        contours.extend(_march_contours(raw_field, ox=ox, oy=oy, lx=lx, ly=ly, nx=nx, ny=ny, name=subvolume.name))
    if not contours:
        raise RealizationError(f"2D geometry {description.name!r} has no interior contours to mesh")

    parent, cell_face = _mesh_box_with_contours(
        contours, ox=ox, oy=oy, lx=lx, ly=ly, h=h, comm=comm, name=description.name
    )
    fields = subvolume_implicit_functions(description)  # priority-resolved; classifies each cell
    tagging = _classify_and_tag(parent, description, fields, cell_face=cell_face, origin=(ox, oy), extent=(lx, ly))
    return parent, tagging


def _realize_2d(description: GeometryDescription, *, h: float, resolution: int | None, comm: MPI.Comm) -> Geometry:
    subvolumes = description.subvolumes
    if not subvolumes:
        raise RealizationError(f"2D geometry {description.name!r} has no subvolumes to realize")
    if len(subvolumes) == 1:
        # A single subvolume is the whole bounding box (VCell's analytic background '1.0'): there is
        # no other region for a complement to occupy, so the expression is moot. A plain structured
        # box mesh, one region, the four box faces named — no internal contours or membranes. This is
        # the single-compartment pattern make_disk_geometry uses: the box mesh *is* the subdomain
        # mesh, so boundary facets index it directly (parent_mesh / tags stay None).
        return _realize_2d_whole_box(description, h=h, comm=comm)

    parent, tagging = _realize_2d_partition(description, h=h, resolution=resolution, comm=comm)
    return _geometry_from_partition(description, parent, tagging, _FACE_NAMES_2D)


def _geometry_from_partition(
    description: GeometryDescription,
    parent: dmesh.Mesh,
    tagging: _Tagging,
    face_names: tuple[str, ...],
) -> Geometry:
    """Assemble a backend :class:`Geometry` from a body-fitted parent mesh and its tagging — the shared
    tail of `_realize_2d` and `_realize_3d`. A volume submesh per region, a surface submesh + boundary
    per realized membrane, and a boundary per named box face (`face_names` orders them axis-major)."""

    tdim = parent.topology.dim
    subdomains: dict[str, SubdomainGeometry] = {}
    for name, tag in tagging.region_tags.items():
        sub_mesh, *_ = dmesh.create_submesh(parent, tdim, tagging.cell_tags.find(tag))
        subdomains[name] = SubdomainGeometry(mesh=sub_mesh, kind="volume")

    # Emptiness is decided globally: create_submesh is collective, and a rank that happens to own none of a
    # membrane's facets must still build it (and name the same boundaries) alongside the others.
    comm = parent.comm
    boundaries: dict[str, BoundaryGeometry] = {}
    for surface in description.surfaces:
        facets = tagging.facet_tags.find(tagging.surface_tags[surface.name])
        if not comm.allreduce(facets.size, op=MPI.SUM):
            continue  # the declared membrane has no realized interface (e.g. regions don't meet)
        membrane_mesh, *_ = dmesh.create_submesh(parent, tdim - 1, facets)
        subdomains[surface.name] = SubdomainGeometry(mesh=membrane_mesh, kind="surface")
        boundaries[surface.name] = BoundaryGeometry(subdomains=(surface.inside, surface.outside), facets=facets)

    for i, face in enumerate(face_names):
        facets = tagging.facet_tags.find(_FACE_TAG_BASE + i)
        if comm.allreduce(facets.size, op=MPI.SUM):
            boundaries[face] = BoundaryGeometry(
                subdomains=tagging.face_regions.get(_FACE_TAG_BASE + i, ()), facets=facets
            )

    return Geometry(
        name=description.name,
        subdomains=subdomains,
        boundaries=boundaries,
        parent_mesh=parent,
        cell_tags=tagging.cell_tags,
        facet_tags=tagging.facet_tags,
    )


def _restrict_to_compartments(
    parent: dmesh.Mesh, tagging: _Tagging, inner_subdomain: str, outer_subdomain: str
) -> tuple[dmesh.Mesh, dmesh.MeshTags, NDArray[np.int32], NDArray[np.int32], dict[str, int]]:
    """Restrict `parent` to the inner + outer compartment cells, dropping the background region — so the
    outer compartment's edge to the (removed) background becomes a true **exterior** boundary of the
    result: a real `ds`-integrable reservoir wall. This reproduces the classic disk-in-annulus geometry
    (parent = disk ∪ annulus, the annulus's outer circle the only external boundary) from the general
    body-fitted partition, gmsh-free — the `background_subdomain` path of `realize_interface_coupled`.

    Returns `(subparent, cell_tags, membrane_facets, wall_facets, region_tags)`: `cell_tags` carries the
    inner/outer region tags on `subparent`, `membrane_facets` are the interior facets straddling the two
    compartments (the shared membrane), `wall_facets` the subparent's exterior (the reservoir wall)."""

    tdim = parent.topology.dim
    inner_tag = tagging.region_tags[inner_subdomain]
    outer_tag = tagging.region_tags[outer_subdomain]
    keep = np.unique(np.concatenate([tagging.cell_tags.find(inner_tag), tagging.cell_tags.find(outer_tag)]))
    subparent, cell_emap, *_ = dmesh.create_submesh(parent, tdim, keep.astype(np.int32))

    # Transfer each subparent cell's region tag from its parent cell (no re-classification needed) — ghost
    # cells included, so a membrane facet on a partition boundary still sees both of its cells.
    sub_map = subparent.topology.index_map(tdim)
    sub_cells = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
    parent_cells = np.asarray(cell_emap.sub_topology_to_topology(sub_cells, False), dtype=np.intp)
    parent_map = parent.topology.index_map(tdim)
    lookup = np.zeros(parent_map.size_local + parent_map.num_ghosts, dtype=np.int32)
    lookup[tagging.cell_tags.indices] = tagging.cell_tags.values
    cell_tag_of = lookup[parent_cells]
    cell_tags = dmesh.meshtags(subparent, tdim, sub_cells, cell_tag_of)

    # Membrane = interior facets straddling an inner and an outer cell; wall = the subparent's exterior.
    subparent.topology.create_connectivity(tdim - 1, tdim)
    f2c = subparent.topology.connectivity(tdim - 1, tdim)
    pair = {int(inner_tag), int(outer_tag)}
    membrane = [
        f
        for f in range(subparent.topology.index_map(tdim - 1).size_local)
        if len(cells := f2c.links(f)) == 2 and {int(cell_tag_of[cells[0]]), int(cell_tag_of[cells[1]])} == pair
    ]
    membrane_facets = np.asarray(membrane, dtype=np.int32)
    wall_facets = dmesh.exterior_facet_indices(subparent.topology)
    return subparent, cell_tags, membrane_facets, wall_facets, {inner_subdomain: inner_tag, outer_subdomain: outer_tag}


def realize_interface_coupled(
    description: GeometryDescription,
    *,
    inner_subdomain: str,
    outer_subdomain: str,
    membrane_subdomain: str,
    interface: str,
    outer: str = "exterior",
    background_subdomain: str | None = None,
    h: float = 0.05,
    resolution: int | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> InterfaceCoupledGeometry:
    """Realize an *imported* two-compartment geometry into an :class:`InterfaceCoupledGeometry` — the
    geometry the bulk-bulk coupled solver (`integrate_interface_coupled`) consumes. Unlike
    :func:`realize` (which builds a plain `Geometry` and discards the submesh entity maps), this keeps
    the three `EntityMap`s relating the two bulk submeshes and the membrane to the shared parent — the
    cross-mesh substrate the coupling form needs (the two-sided trace on the parent's interface `dS`).

    So a cross-compartment model goes end-to-end through the real pipeline: VCell geometry →
    `import_geometry` → `realize_interface_coupled` → `integrate_interface_coupled`, with no
    hand-built parallel mesh. `inner_subdomain` / `outer_subdomain` name the two volume compartments
    and `membrane_subdomain` the SurfaceClass between them; `interface` is the boundary label.

    **The reservoir wall** (`outer`) is where a coupled model may hold a Dirichlet. By default the outer
    compartment is box-bounded and the wall is the whole box exterior. Pass `background_subdomain` (the
    name of a third, complement subvolume surrounding the outer compartment) to instead **bound** the
    geometry to inner + outer: the outer compartment's outer edge (its interface to the dropped
    background) becomes the exterior wall — a true circular reservoir edge, reproducing the classic
    disk-in-annulus geometry gmsh-free (`_restrict_to_compartments`).
    """

    expected = {inner_subdomain, outer_subdomain}
    volume_names = {sv.name for sv in description.subvolumes}
    if not expected <= volume_names:
        raise RealizationError(
            f"interface-coupled realization needs both compartments {sorted(expected)} among the "
            f"geometry's subvolumes {sorted(volume_names)}"
        )
    if background_subdomain is not None and background_subdomain not in volume_names:
        raise RealizationError(
            f"background_subdomain {background_subdomain!r} is not among the geometry's subvolumes "
            f"{sorted(volume_names)}"
        )
    surface = next((s for s in description.surfaces if s.name == membrane_subdomain), None)
    if surface is None:
        raise RealizationError(
            f"interface-coupled realization needs a SurfaceClass named {membrane_subdomain!r}; "
            f"geometry {description.name!r} has {[s.name for s in description.surfaces]}"
        )

    if description.dim == 3:
        parent, tagging = _realize_3d_partition(description, h=h, resolution=resolution, comm=comm)
    else:
        parent, tagging = _realize_2d_partition(description, h=h, resolution=resolution, comm=comm)
    tdim = parent.topology.dim
    interface_tag = tagging.surface_tags[membrane_subdomain]

    if background_subdomain is not None:
        # Bounded (annulus) parent: drop the background so the outer compartment's outer edge is a real
        # exterior `ds` reservoir wall; membrane and cell tags are re-derived on the restricted mesh.
        parent, cell_tags, interface_facets, wall_facets, region_tags = _restrict_to_compartments(
            parent, tagging, inner_subdomain, outer_subdomain
        )
    else:
        # Box-bounded parent: the reservoir wall is the outer compartment's share of the box exterior.
        cell_tags = tagging.cell_tags
        region_tags = tagging.region_tags
        interface_facets = tagging.facet_tags.find(interface_tag)
        parent.topology.create_connectivity(tdim - 1, tdim)
        wall_facets = dmesh.exterior_facet_indices(parent.topology)
    # Only the outer compartment's exterior is the reservoir wall: an image's inner compartment (or a third
    # region) may touch the box too, and its exterior facets are not the outer reservoir's boundary.
    wall_facets = _facets_of_region(parent, cell_tags, wall_facets, region_tags[outer_subdomain])
    if not comm.allreduce(interface_facets.size, op=MPI.SUM):  # global: every rank raises, or none does
        raise RealizationError(f"the membrane {membrane_subdomain!r} has no realized interface facets")

    # Retain the entity maps (realize() discards them) — they relate each submesh to the parent so the
    # coupling form on the parent's interface dS can pull in both bulk traces.
    inner_mesh, inner_emap, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(region_tags[inner_subdomain]))
    outer_mesh, outer_emap, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(region_tags[outer_subdomain]))
    membrane_mesh, membrane_emap, *_ = dmesh.create_submesh(parent, tdim - 1, interface_facets)

    # A single merged facet tagging: the membrane at `interface_tag` (coupling `dS`) and the reservoir
    # wall at `_OUTER_WALL_TAG` (`ds`). The two facet sets are disjoint (interior membrane vs exterior
    # wall), so the tags are unambiguous; `integrate_interface_coupled` reads `interface_tag`/`outer_tag`.
    idx = np.concatenate([interface_facets, wall_facets]).astype(np.int32)
    val = np.concatenate(
        [
            np.full(interface_facets.size, interface_tag, dtype=np.int32),
            np.full(wall_facets.size, _OUTER_WALL_TAG, dtype=np.int32),
        ]
    )
    order = np.argsort(idx)
    facet_tags = dmesh.meshtags(parent, tdim - 1, idx[order], val[order])

    return InterfaceCoupledGeometry(
        name=description.name,
        inner_subdomain=inner_subdomain,
        outer_subdomain=outer_subdomain,
        membrane_subdomain=membrane_subdomain,
        inner_mesh=inner_mesh,
        outer_mesh=outer_mesh,
        membrane_mesh=membrane_mesh,
        inner_entity_map=inner_emap,
        outer_entity_map=outer_emap,
        membrane_entity_map=membrane_emap,
        parent_mesh=parent,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        inner_region_tag=region_tags[inner_subdomain],
        outer_region_tag=region_tags[outer_subdomain],
        interface=interface,
        interface_tag=interface_tag,
        outer=outer,
        outer_tag=_OUTER_WALL_TAG,
    )


def _realize_2d_whole_box(description: GeometryDescription, *, h: float, comm: MPI.Comm) -> Geometry:
    """Realize a single-subvolume 2D geometry as a plain structured box: one `volume` region over the
    whole `extent` / `origin` rectangle, with the four box faces (`x_minus` … `y_plus`) named for
    boundary conditions. Mirrors :func:`make_disk_geometry` — the box mesh is the subdomain mesh."""

    subvolume = description.subvolumes[0]
    ox, oy = description.origin[0], description.origin[1]
    lx, ly = description.extent[0], description.extent[1]
    nx, ny = max(1, round(lx / h)), max(1, round(ly / h))
    box = dmesh.create_rectangle(
        comm,
        [np.array([ox, oy]), np.array([ox + lx, oy + ly])],
        [nx, ny],
        dmesh.CellType.triangle,
    )
    fdim = box.topology.dim - 1
    markers = (
        ("x_minus", lambda p: np.isclose(p[0], ox)),
        ("x_plus", lambda p: np.isclose(p[0], ox + lx)),
        ("y_minus", lambda p: np.isclose(p[1], oy)),
        ("y_plus", lambda p: np.isclose(p[1], oy + ly)),
    )
    boundaries: dict[str, BoundaryGeometry] = {}
    for face, marker in markers:
        facets = dmesh.locate_entities_boundary(box, fdim, marker)
        if facets.size:
            boundaries[face] = BoundaryGeometry(subdomains=(subvolume.name,), facets=facets)

    subdomains = {subvolume.name: SubdomainGeometry(mesh=box, kind="volume")}
    return Geometry(name=description.name, subdomains=subdomains, boundaries=boundaries)


def _march_contours(
    field: Expr, *, ox: float, oy: float, lx: float, ly: float, nx: int, ny: int, name: str
) -> list[NDArray[np.float64]]:
    """Sample ``field`` over the box on an ``nx × ny`` grid and march every ``φ = 0`` contour, each
    returned as physical (x, y) vertices. ``nx`` / ``ny`` are per-axis point counts (so the spacing is
    ≈ h on each axis even for a non-square box). v1 requires each contour to be strictly inside the box
    (a subvolume touching the box boundary is a later slice)."""

    xs = np.linspace(ox, ox + lx, nx)
    ys = np.linspace(oy, oy + ly, ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    phi = _eval_field(field, (grid_x, grid_y))

    dx, dy = lx / (nx - 1), ly / (ny - 1)
    out: list[NDArray[np.float64]] = []
    for rows_cols in find_contours(phi, 0.0):
        xy = np.column_stack([ox + rows_cols[:, 1] * dx, oy + rows_cols[:, 0] * dy])
        on_edge = (
            np.isclose(xy[:, 0], ox)
            | np.isclose(xy[:, 0], ox + lx)
            | np.isclose(xy[:, 1], oy)
            | np.isclose(xy[:, 1], oy + ly)
        )
        if on_edge.any():
            raise NotImplementedError(
                f"2D realization requires the boundary of {name!r} to be strictly inside the box "
                "(a subvolume touching the box boundary is a later slice)"
            )
        out.append(xy)
    return out


_NetgenArrays = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int32]]


def _mesh_on_rank0(
    comm: MPI.Comm, build: Callable[[], _NetgenArrays], *, cell: str, gdim: int
) -> tuple[dmesh.Mesh, dmesh.MeshTags]:
    """Run the serial Netgen ``build`` on rank 0 only and distribute its mesh through DOLFINx, returning
    the mesh and its cell→material tag.

    ``create_mesh`` reads each rank's ``cells``/``points`` as that rank's *share* of one global input, so
    every rank handing over the full Netgen mesh builds overlapping copies and indexes ``material`` out
    of range (the partition path crashed under ``mpiexec -n 2`` before this). Rank 0 alone supplies the
    input; the others pass empty arrays. ``original_cell_index`` then indexes rank 0's arrays, so the
    per-cell material is broadcast for every rank to realign its local cells. A failure inside ``build``
    is broadcast and re-raised on every rank rather than leaving the others blocked in a collective.
    """

    nverts = {"triangle": 3, "tetrahedron": 4}[cell]
    points = np.empty((0, gdim), dtype=np.float64)
    cells = np.empty((0, nverts), dtype=np.int64)
    shared: NDArray[np.int32] | BaseException | None = None
    if comm.rank == 0:
        try:
            points, cells, shared = build()
        except Exception as exc:  # re-raised on every rank below
            shared = exc
    shared = comm.bcast(shared, root=0)
    if isinstance(shared, BaseException):
        raise shared
    assert shared is not None
    material = shared

    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 1, shape=(gdim,)))
    # Ghost across shared facets: a membrane is an *interior* facet set, so in parallel both cells
    # beside a partition-boundary membrane facet must be local — for tagging it here and for the `dS`
    # interface integrals the coupled solvers assemble on it (serial runs have no ghosts either way).
    partitioner = dmesh.create_cell_partitioner(dmesh.GhostMode.shared_facet)
    mesh = dmesh.create_mesh(comm, cells, domain, points, partitioner=partitioner)

    # create_mesh may reorder cells for locality; realign the per-cell material via the input index —
    # for ghost cells too, so a neighbour across the partition boundary carries its region.
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    n_all = cell_map.size_local + cell_map.num_ghosts
    all_cells = np.arange(n_all, dtype=np.int32)
    cell_material = material[np.asarray(mesh.topology.original_cell_index)[:n_all]]
    cell_face = dmesh.meshtags(mesh, tdim, all_cells, cell_material)
    return mesh, cell_face


def _mesh_box_with_contours(
    contours: list[NDArray[np.float64]],
    *,
    ox: float,
    oy: float,
    lx: float,
    ly: float,
    h: float,
    comm: MPI.Comm,
    name: str,
) -> tuple[dmesh.Mesh, dmesh.MeshTags]:
    """The body-fitted DOLFINx mesh of the box with every ``contours`` polyline as a conforming internal
    boundary, **and** its cell→region tag — :func:`_netgen_box_with_contours` meshed on rank 0 and
    distributed by :func:`_mesh_on_rank0`."""

    return _mesh_on_rank0(
        comm,
        lambda: _netgen_box_with_contours(contours, ox=ox, oy=oy, lx=lx, ly=ly, h=h, name=name),
        cell="triangle",
        gdim=2,
    )


def _netgen_box_with_contours(
    contours: list[NDArray[np.float64]],
    *,
    ox: float,
    oy: float,
    lx: float,
    ly: float,
    h: float,
    name: str,
) -> _NetgenArrays:
    """Build a Netgen model of the box with every ``contours`` polyline embedded as a conforming
    internal boundary, returning its points, cells and per-cell material index (serial). Netgen assigns
    a material index per element (``el.index``) from the leftdomain/rightdomain of the embedded curves;
    that index is one owner per body-fit region — exactly the tag :func:`_classify_and_tag` needs to
    keep each region whole. The containment forest of the (non-intersecting) contours supplies those
    domains: a top-level contour separates its interior from the box background, a nested one from its
    parent's interior (nested shapes nest, disjoint shapes sit side by side). LGPL Netgen replaces GPL
    gmsh here (ADR 008); region / membrane / face tagging still happens afterwards in DOLFINx."""

    # Simplify (Douglas–Peucker) and orient each contour CCW so its interior is on the left — Netgen's
    # ``leftdomain``. The tolerance bounds how far the simplified polygon may deviate from the marched
    # contour; it is deliberately *tight* (0.01·h) because the deviation is a systematic *inward* bias
    # (chords cut inside a convex arc), so a loose tolerance shrinks a curved region's area measurably
    # (0.25·h cost ~2.6% on a disk) while — since Netgen resamples the boundary at h regardless — buying
    # no reduction in mesh size. 0.01·h keeps body-fit geometry within ~0.1% of the analytic shape, the
    # fidelity gmsh's exact-curve boundary gives, at the same node count. (See the migration notes.)
    polys: list[NDArray[np.float64]] = []
    for contour in contours:
        verts = approximate_polygon(contour, tolerance=0.01 * h)
        if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        if len(verts) < 3:
            raise RealizationError(f"a contour of {name!r} degenerated to {len(verts)} vertices")
        if _signed_area(verts) < 0.0:
            verts = verts[::-1]
        polys.append(verts)

    parents = _containment_parents(polys)
    interior_domain = [i + 2 for i in range(len(polys))]  # the box background is domain 1

    geo = SplineGeometry()
    corners = [
        geo.AppendPoint(ox, oy),
        geo.AppendPoint(ox + lx, oy),
        geo.AppendPoint(ox + lx, oy + ly),
        geo.AppendPoint(ox, oy + ly),
    ]
    for i in range(4):
        geo.Append(["line", corners[i], corners[(i + 1) % 4]], leftdomain=1, rightdomain=0)
    for i, poly in enumerate(polys):
        outside = 1 if parents[i] < 0 else interior_domain[parents[i]]
        pids = [geo.AppendPoint(float(x), float(y)) for x, y in poly]
        for j in range(len(poly)):
            geo.Append(
                ["line", pids[j], pids[(j + 1) % len(poly)]],
                leftdomain=interior_domain[i],
                rightdomain=outside,
            )
    ngmesh = geo.GenerateMesh(maxh=float(h))

    points = np.array([list(p.p)[:2] for p in ngmesh.Points()], dtype=np.float64)
    cells = np.array([[v.nr - 1 for v in el.vertices] for el in ngmesh.Elements2D()], dtype=np.int64)
    material = np.array([el.index for el in ngmesh.Elements2D()], dtype=np.int32)
    return points, cells, material


def _signed_area(poly: NDArray[np.float64]) -> float:
    """Shoelace signed area of the closed polygon `poly` (CCW positive)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_polygon(pt: NDArray[np.float64], poly: NDArray[np.float64]) -> bool:
    """Ray-casting parity test: is `pt` inside the closed polygon `poly`?"""
    x, y = float(pt[0]), float(pt[1])
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, -1), np.roll(ys, -1)
    crosses = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-300) + xs)
    return bool(np.count_nonzero(crosses) % 2 == 1)


def _containment_parents(polys: list[NDArray[np.float64]]) -> list[int]:
    """Immediate-parent index for each polygon — the smallest-area polygon that contains it, or -1 if
    top-level. `_march_contours` yields non-intersecting contours, so one vertex of a polygon lies
    inside another iff the whole polygon does: a single point test per pair suffices."""
    areas = [abs(_signed_area(p)) for p in polys]
    parents: list[int] = []
    for i, poly in enumerate(polys):
        probe = poly[0]
        best, best_area = -1, np.inf
        for j, other in enumerate(polys):
            if j == i or areas[j] < areas[i]:
                continue  # a smaller-or-equal polygon cannot strictly contain this one
            if _point_in_polygon(probe, other) and areas[j] < best_area:
                best, best_area = j, areas[j]
        parents.append(best)
    return parents


# -- dim 3: body-fitted, box-partitioned ----------------------------------------


def _realize_3d(description: GeometryDescription, *, h: float, resolution: int | None, comm: MPI.Comm) -> Geometry:
    subvolumes = description.subvolumes
    if not subvolumes:
        raise RealizationError(f"3D geometry {description.name!r} has no subvolumes to realize")
    if len(subvolumes) == 1:
        return _realize_3d_whole_box(description, h=h, comm=comm)
    parent, tagging = _realize_3d_partition(description, h=h, resolution=resolution, comm=comm)
    return _geometry_from_partition(description, parent, tagging, _FACE_NAMES_3D)


def _realize_3d_whole_box(description: GeometryDescription, *, h: float, comm: MPI.Comm) -> Geometry:
    """A single-subvolume 3D geometry is the bounding box itself: a structured tetrahedral box mesh, one
    ``volume`` region, the six box faces named. The 3D analog of :func:`_realize_2d_whole_box`."""

    subvolume = description.subvolumes[0]
    ox, oy, oz = description.origin[0], description.origin[1], description.origin[2]
    lx, ly, lz = description.extent[0], description.extent[1], description.extent[2]
    counts = [max(1, round(length / h)) for length in (lx, ly, lz)]
    box = dmesh.create_box(
        comm,
        [np.array([ox, oy, oz]), np.array([ox + lx, oy + ly, oz + lz])],
        counts,
        dmesh.CellType.tetrahedron,
    )
    fdim = box.topology.dim - 1
    markers = (
        ("x_minus", lambda p: np.isclose(p[0], ox)),
        ("x_plus", lambda p: np.isclose(p[0], ox + lx)),
        ("y_minus", lambda p: np.isclose(p[1], oy)),
        ("y_plus", lambda p: np.isclose(p[1], oy + ly)),
        ("z_minus", lambda p: np.isclose(p[2], oz)),
        ("z_plus", lambda p: np.isclose(p[2], oz + lz)),
    )
    boundaries: dict[str, BoundaryGeometry] = {}
    for face, marker in markers:
        facets = dmesh.locate_entities_boundary(box, fdim, marker)
        if facets.size:
            boundaries[face] = BoundaryGeometry(subdomains=(subvolume.name,), facets=facets)
    subdomains = {subvolume.name: SubdomainGeometry(mesh=box, kind="volume")}
    return Geometry(name=description.name, subdomains=subdomains, boundaries=boundaries)


def _realize_3d_partition(
    description: GeometryDescription, *, h: float, resolution: int | None, comm: MPI.Comm
) -> tuple[dmesh.Mesh, _Tagging]:
    """Body-fit and tag a multi-subvolume 3D geometry: marching-cubes each analytic subvolume's implicit
    surface, mesh the box partitioned by them (`_mesh_box_with_surfaces`), then classify each cell by the
    priority-resolved fields (`_classify_and_tag`, shared with 2D). The 3D analog of
    `_realize_2d_partition`."""

    if _is_image(description):
        return _realize_image_partition(description, h=h, comm=comm)
    subvolumes = description.subvolumes
    for subvolume in subvolumes[:-1]:
        if subvolume.type != "analytic" or subvolume.expression is None:
            raise RealizationError(
                f"3D geometry {description.name!r} subvolume {subvolume.name!r} must be 'analytic' with an "
                f"expression to be realized (got type {subvolume.type!r})"
            )

    ox, oy, oz = description.origin[0], description.origin[1], description.origin[2]
    lx, ly, lz = description.extent[0], description.extent[1], description.extent[2]
    origin, extent = (ox, oy, oz), (lx, ly, lz)
    counts: tuple[int, int, int]
    if resolution is None:
        counts = (
            min(257, max(33, round(lx / h) + 1)),
            min(257, max(33, round(ly / h) + 1)),
            min(257, max(33, round(lz / h) + 1)),
        )
    else:
        counts = (resolution, resolution, resolution)

    raw_fields: list[Expr] = []
    for subvolume in subvolumes[:-1]:
        assert subvolume.expression is not None  # checked above
        raw_fields.append(lower_predicate(parse(subvolume.expression)))
    surfaces = [
        _march_surface(field, origin=origin, extent=extent, counts=counts, name=sv.name)
        for field, sv in zip(raw_fields, subvolumes[:-1], strict=True)
    ]
    parents = _surface_containment(raw_fields, origin=origin, extent=extent, counts=counts)

    parent, cell_face = _mesh_box_with_surfaces(
        surfaces, parents, origin=origin, extent=extent, h=h, comm=comm, name=description.name
    )
    fields = subvolume_implicit_functions(description)
    tagging = _classify_and_tag(parent, description, fields, cell_face=cell_face, origin=origin, extent=extent)
    return parent, tagging


def _march_surface(
    field: Expr,
    *,
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    counts: tuple[int, int, int],
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sample ``field`` on the box grid and marching-cubes its ``φ = 0`` isosurface, returned as physical
    (verts, triangle faces). v1 requires the surface strictly inside the box (a subvolume touching the box
    boundary is a later slice), mirroring :func:`_march_contours`."""

    ox, oy, oz = origin
    lx, ly, lz = extent
    nx, ny, nz = counts
    grid = np.meshgrid(
        np.linspace(ox, ox + lx, nx), np.linspace(oy, oy + ly, ny), np.linspace(oz, oz + lz, nz), indexing="ij"
    )
    phi = _eval_field(field, (grid[0], grid[1], grid[2]))
    spacing = (lx / (nx - 1), ly / (ny - 1), lz / (nz - 1))
    try:
        verts, faces, _, _ = marching_cubes(phi, 0.0, spacing=spacing)
    except (ValueError, RuntimeError) as exc:
        raise RealizationError(f"subvolume {name!r} has no φ = 0 isosurface inside the box") from exc
    verts = verts + np.array([ox, oy, oz])
    on_edge = (
        np.isclose(verts[:, 0], ox)
        | np.isclose(verts[:, 0], ox + lx)
        | np.isclose(verts[:, 1], oy)
        | np.isclose(verts[:, 1], oy + ly)
        | np.isclose(verts[:, 2], oz)
        | np.isclose(verts[:, 2], oz + lz)
    )
    if bool(on_edge.any()):
        raise NotImplementedError(
            f"3D realization requires the boundary of {name!r} to be strictly inside the box "
            "(a subvolume touching the box boundary is a later slice)"
        )
    return verts.astype(np.float64), faces.astype(np.int64)


def _surface_containment(
    raw_fields: list[Expr],
    *,
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    counts: tuple[int, int, int],
) -> list[int]:
    """Immediate-parent subvolume index for each subvolume — the smallest enclosing one, or -1 (box).
    Computed from the fields (not the marched geometry): subvolume ``i`` is inside ``j`` iff ``j``'s field
    is negative at ``i``'s deepest interior grid point; the parent is the smallest such ``j``. The 3D
    analog of :func:`_containment_parents`."""

    ox, oy, oz = origin
    lx, ly, lz = extent
    nx, ny, nz = counts
    grid = np.meshgrid(
        np.linspace(ox, ox + lx, nx), np.linspace(oy, oy + ly, ny), np.linspace(oz, oz + lz, nz), indexing="ij"
    )
    coords = (grid[0].ravel(), grid[1].ravel(), grid[2].ravel())
    phis = [_eval_field(f, coords) for f in raw_fields]
    inside = [phi < 0 for phi in phis]
    sizes = [int(m.sum()) for m in inside]
    parents: list[int] = []
    for i, phi_i in enumerate(phis):
        if not bool(inside[i].any()):
            parents.append(-1)
            continue
        rep = int(np.argmin(phi_i))  # deepest interior point of subvolume i
        best, best_size = -1, np.inf
        for j in range(len(phis)):
            if j == i or sizes[j] < sizes[i]:
                continue  # a smaller-or-equal subvolume cannot enclose this one
            if bool(inside[j][rep]) and sizes[j] < best_size:
                best, best_size = j, sizes[j]
        parents.append(best)
    return parents


def _mesh_box_with_surfaces(
    surfaces: list[tuple[NDArray[np.float64], NDArray[np.int64]]],
    parents: list[int],
    *,
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    h: float,
    comm: MPI.Comm,
    name: str,
) -> tuple[dmesh.Mesh, dmesh.MeshTags]:
    """The body-fitted DOLFINx tetrahedral mesh of the box partitioned by the marched ``surfaces``,
    **and** its cell→region tag — :func:`_netgen_box_with_surfaces` meshed on rank 0 and distributed by
    :func:`_mesh_on_rank0`."""

    return _mesh_on_rank0(
        comm,
        lambda: _netgen_box_with_surfaces(surfaces, parents, origin=origin, extent=extent, h=h, name=name),
        cell="tetrahedron",
        gdim=3,
    )


def _netgen_box_with_surfaces(
    surfaces: list[tuple[NDArray[np.float64], NDArray[np.int64]]],
    parents: list[int],
    *,
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    h: float,
    name: str,
) -> _NetgenArrays:
    """Volume-mesh the box partitioned by the marched ``surfaces`` (serial), returning its points,
    tetrahedra and per-cell region index. The recipe (ADR 008 §8, the 3D analog of the 2D
    `SplineGeometry` leftdomain/rightdomain path): the box surface is meshed by Netgen's CSG (it carries
    the local mesh-size function that raw triangles lack); each marched surface is **re-meshed** through
    `STLGeometry` (raw marched triangles as an interface produce slivers); the surfaces are merged into one
    `Mesh` with a `FaceDescriptor` per surface whose ``domin``/``domout`` come from the containment forest;
    `GenerateVolumeMesh` fills the domains and ``el.index`` is the per-region tag."""

    ox, oy, oz = origin
    lx, ly, lz = extent
    interior_domain = [i + 2 for i in range(len(surfaces))]  # box background is domain 1

    box_geo = CSGeometry()
    box_geo.Add(OrthoBrick(CsgPnt(ox, oy, oz), CsgPnt(ox + lx, oy + ly, oz + lz)))
    box_mesh = box_geo.GenerateMesh(maxh=h)

    merged = NetgenMesh(dim=3)
    _copy_surface(merged, box_mesh, merged.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1)))
    for i, (verts, faces) in enumerate(surfaces):
        outside = 1 if parents[i] < 0 else interior_domain[parents[i]]
        fd = merged.Add(FaceDescriptor(surfnr=i + 2, domin=interior_domain[i], domout=outside, bc=i + 2))
        _copy_surface(merged, _remesh_surface(verts, faces, h), fd)
    merged.GenerateVolumeMesh()

    points = np.array([list(p.p) for p in merged.Points()], dtype=np.float64)
    cells = np.array([[v.nr - 1 for v in el.vertices] for el in merged.Elements3D()], dtype=np.int64)
    material = np.array([el.index for el in merged.Elements3D()], dtype=np.int32)
    return points, cells, material


def _copy_surface(merged: NetgenMesh, source: NetgenMesh, fd: int) -> None:
    """Copy `source`'s surface triangles into `merged` under face descriptor `fd`, sharing points so the
    merged surface stays conforming (the Netgen merge idiom)."""

    pmap: dict[object, object] = {}
    for element in source.Elements2D():
        for vertex in element.vertices:
            if vertex not in pmap:
                pmap[vertex] = merged.Add(source[vertex])
    for element in source.Elements2D():
        merged.Add(Element2D(fd, [pmap[v] for v in element.vertices]))  # type: ignore[misc]


def _remesh_surface(verts: NDArray[np.float64], faces: NDArray[np.int64], h: float) -> NetgenMesh:
    """Re-mesh a marched triangle surface to a quality triangulation via Netgen's STL surface mesher
    (using the raw marched triangles directly as a volume-mesh interface produces slivers)."""

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.stl"
        _write_stl(path, verts, faces)
        return STLGeometry(str(path)).GenerateMesh(maxh=h)


def _write_stl(path: Path, verts: NDArray[np.float64], faces: NDArray[np.int64]) -> None:
    """Write an ASCII STL for the triangle surface (verts, faces) with per-facet normals."""

    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    lines = ["solid s"]
    for i in range(faces.shape[0]):
        lines.append(f"facet normal {normals[i, 0]:.6e} {normals[i, 1]:.6e} {normals[i, 2]:.6e}")
        lines.append(" outer loop")
        for p in (a[i], b[i], c[i]):
            lines.append(f"  vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append(" endloop")
        lines.append("endfacet")
    lines.append("endsolid s")
    path.write_text("\n".join(lines))


def _classify_and_tag(
    parent: dmesh.Mesh,
    description: GeometryDescription,
    fields: dict[str, Expr],
    *,
    cell_face: dmesh.MeshTags,
    origin: tuple[float, ...],
    extent: tuple[float, ...],
) -> _Tagging:
    """Tag the body-fitted mesh by the priority-resolved implicit fields. Each cell is provisionally
    assigned to the subvolume whose `φ` is negative at its midpoint (the fields partition the plane, so
    exactly one is negative), then each fragment FACE is given its **majority** vote: a face is one
    body-fit region, so resolving it as a whole keeps a boundary cell — body-fit outside the simplified
    contour but with its midpoint still inside the exact shape — from being mis-assigned across the
    boundary (which would bulge the region boundary by ~½ a cell). A membrane facet (interior, between two
    regions) is named by the `SurfaceClass` for that pair; box faces are exterior facets, classified by
    position and the region they touch."""

    return _tag_facets(
        parent, description, _vote_cell_values(parent, description, fields, cell_face), origin=origin, extent=extent
    )


def _vote_cell_values(
    parent: dmesh.Mesh, description: GeometryDescription, fields: dict[str, Expr], cell_face: dmesh.MeshTags
) -> NDArray[np.int32]:
    """Each cell's region tag (owned and ghost cells) by the priority-resolved analytic fields, each
    body-fit face resolved as a whole by its majority (see :func:`_classify_and_tag`)."""

    subvolumes = description.subvolumes
    region_tags = {sv.name: i + 1 for i, sv in enumerate(subvolumes)}
    tdim = parent.topology.dim
    comm = parent.comm

    # Owned and ghost cells alike: a ghost's region is needed to classify the partition-boundary facets.
    gdim = parent.geometry.dim
    cell_map = parent.topology.index_map(tdim)
    n_owned = cell_map.size_local
    cells = np.arange(n_owned + cell_map.num_ghosts, dtype=np.int32)
    cell_mid = dmesh.compute_midpoints(parent, tdim, cells)
    per_cell = np.zeros(cells.size, dtype=np.int32)
    for name, tag in region_tags.items():
        per_cell[_eval_field(fields[name], tuple(cell_mid[:, k] for k in range(gdim))) < 0] = tag
    per_cell[per_cell == 0] = region_tags[subvolumes[-1].name]  # any unclaimed cell → background

    # Resolve each body-fit face as a whole by the majority of its cells' votes (consistent with the
    # conforming boundary, vs a per-cell midpoint test that frays it). The vote is global — owned cells
    # counted once across ranks — so a face split by the partition resolves the same way everywhere; ties
    # go to the lowest tag, as `np.bincount(...).argmax()` did serially.
    face_of_cell = np.zeros(cells.size, dtype=np.int64)
    face_of_cell[cell_face.indices] = cell_face.values
    n_faces = comm.allreduce(int(face_of_cell.max(initial=0)) + 1, op=MPI.MAX)
    votes = np.zeros((n_faces, len(subvolumes) + 1), dtype=np.int64)
    np.add.at(votes, (face_of_cell[:n_owned], per_cell[:n_owned]), 1)
    comm.Allreduce(MPI.IN_PLACE, votes, op=MPI.SUM)
    return np.asarray(votes.argmax(axis=1).astype(np.int32)[face_of_cell], dtype=np.int32)


def _tag_facets(
    parent: dmesh.Mesh,
    description: GeometryDescription,
    cell_values: NDArray[np.int32],
    *,
    origin: tuple[float, ...],
    extent: tuple[float, ...],
) -> _Tagging:
    """The cell and facet tagging of a partitioned mesh whose cells (owned and ghost) carry region tags
    ``cell_values`` (subvolume index + 1): membrane facets (interior, between the two subvolumes of a
    declared SurfaceClass) and box-face facets (exterior, classified by position), vectorized."""

    subvolumes = description.subvolumes
    region_tags = {sv.name: i + 1 for i, sv in enumerate(subvolumes)}
    tag_to_name = {tag: name for name, tag in region_tags.items()}
    tdim = parent.topology.dim
    comm = parent.comm
    cells = np.arange(cell_values.size, dtype=np.int32)
    cell_tags = dmesh.meshtags(parent, tdim, cells, cell_values)

    surface_tags = {sc.name: _SURFACE_TAG_BASE + i for i, sc in enumerate(description.surfaces)}
    # a (left, right) region-tag lookup → membrane tag (0: no declared SurfaceClass for that pair)
    n_tags = len(subvolumes) + 1
    pair_tag = np.zeros((n_tags, n_tags), dtype=np.int32)
    for i, sc in enumerate(description.surfaces):
        a, b = region_tags.get(sc.inside), region_tags.get(sc.outside)
        if a is not None and b is not None:
            pair_tag[a, b] = pair_tag[b, a] = _SURFACE_TAG_BASE + i

    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    num_facets = parent.topology.index_map(tdim - 1).size_local
    offsets = np.asarray(facet_to_cell.offsets[: num_facets + 1])
    links = np.asarray(facet_to_cell.array)
    count = np.diff(offsets)
    first = links[offsets[:-1]]
    second = np.where(count == 2, links[np.minimum(offsets[:-1] + 1, links.size - 1)], first)

    # exterior facets: the box face their midpoint lies on (axis-major, the first match — as _classify_face)
    exterior = np.flatnonzero(count == 1)
    mids = dmesh.compute_midpoints(parent, tdim - 1, exterior.astype(np.int32))
    face = np.full(exterior.size, -1, dtype=np.int64)
    for axis in range(len(origin)):
        for side, value in ((0, origin[axis]), (1, origin[axis] + extent[axis])):
            hit = (face < 0) & np.isclose(mids[:, axis], value)
            face[hit] = 2 * axis + side
    on_face = face >= 0
    box_facets = exterior[on_face]
    box_tags = (_FACE_TAG_BASE + face[on_face]).astype(np.int32)

    # interior facets between two regions with a declared SurfaceClass: membranes
    interior = np.flatnonzero(count == 2)
    membrane = pair_tag[cell_values[first[interior]], cell_values[second[interior]]]
    named = membrane > 0
    idx = np.concatenate([box_facets, interior[named]]).astype(np.int32)
    val = np.concatenate([box_tags, membrane[named]]).astype(np.int32)
    order = np.argsort(idx)
    facet_tags = dmesh.meshtags(parent, tdim - 1, idx[order], val[order])

    face_regions: dict[int, set[str]] = {}
    box_cells = cell_values[first[box_facets]]
    for tag in np.unique(box_tags).tolist():
        face_regions[int(tag)] = {tag_to_name[int(v)] for v in np.unique(box_cells[box_tags == tag]).tolist()}
    # Which regions touch each box face is a property of the whole mesh, not of one rank's share.
    merged: dict[int, set[str]] = {}
    for part in comm.allgather(face_regions):
        for tag, names in part.items():
            merged.setdefault(tag, set()).update(names)
    return _Tagging(
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        region_tags=region_tags,
        surface_tags=surface_tags,
        face_regions={tag: tuple(sorted(names)) for tag, names in merged.items()},
    )


def _classify_face(point: tuple[float, ...], *, origin: tuple[float, ...], extent: tuple[float, ...]) -> int | None:
    """Index into the box face names for a facet centroid on a box face, else ``None`` (a membrane).
    Faces are ordered axis-major to match ``_FACE_NAMES_2D`` / ``_FACE_NAMES_3D``: ``2*axis`` is
    ``<axis>_minus``, ``2*axis + 1`` is ``<axis>_plus`` (x, then y, then z)."""

    for axis, (o, length, c) in enumerate(zip(origin, extent, point, strict=True)):
        if np.isclose(c, o):
            return 2 * axis
        if np.isclose(c, o + length):
            return 2 * axis + 1
    return None


# -- image geometries: label field → conforming boundaries → Netgen -------------------------------


def _is_image(description: GeometryDescription) -> bool:
    return any(subvolume.type == "image" for subvolume in description.subvolumes)


def _realize_image_partition(
    description: GeometryDescription, *, h: float, comm: MPI.Comm
) -> tuple[dmesh.Mesh, _Tagging]:
    """Body-fit and tag an image geometry (2D or 3D; any topology — nested, touching the box, junctions):
    its smoothed label field (:func:`~vcell_fenics.backend.labels.label_geometry`), the conforming
    boundaries between its subvolumes (:func:`~vcell_fenics.backend.label_surfaces.extract_boundary`),
    and a Netgen mesh with those boundaries embedded, built on rank 0. Netgen's domain *is* the region
    (subvolume index + 1), so no classification is needed — only the facet tagging. Topology changes at
    this ``h`` are reported as :class:`ImageGeometryWarning`."""

    dim = description.dim
    origin = tuple(float(description.origin[i]) for i in range(dim))
    extent = tuple(float(description.extent[i]) for i in range(dim))
    if dim == 3:
        estimate = 6.0 * np.sqrt(2.0) * float(np.prod(extent)) / h**3
        if estimate > _MAX_IMAGE_TETS:
            coarsest = (6.0 * np.sqrt(2.0) * float(np.prod(extent)) / _MAX_IMAGE_TETS) ** (1.0 / 3.0)
            raise RealizationError(
                f"image geometry {description.name!r} at h = {h:g} would need ~{estimate:.2g} tetrahedra "
                f"(the limit is {_MAX_IMAGE_TETS:.2g}); use h ≥ {coarsest:.3g}"
            )
    notes: list[str] = []

    def build() -> _NetgenArrays:
        labels = label_geometry(description, h=h)
        notes.extend(labels.warnings)
        boundary = extract_boundary(labels.grid, extent=extent)
        return _netgen_from_curves(boundary, h) if dim == 2 else _netgen_from_surfaces(boundary, h)

    parent, material = _mesh_on_rank0(comm, build, cell="triangle" if dim == 2 else "tetrahedron", gdim=dim)
    for note in comm.bcast(notes, root=0):
        warnings.warn(f"image geometry {description.name!r}: {note}", ImageGeometryWarning, stacklevel=3)
    cell_values = np.zeros(material.indices.size, dtype=np.int32)
    cell_values[material.indices] = material.values
    return parent, _tag_facets(parent, description, cell_values, origin=origin, extent=extent)


def _domains(boundary: LabelBoundary) -> dict[int, int]:
    """Netgen domain numbers for the regions a boundary separates: contiguous from 1 (0 is outside the
    box), in region order."""

    regions = sorted({int(side) for side in boundary.pairs.ravel().tolist() if side >= 0})
    return {region: k + 1 for k, region in enumerate(regions)}


def _netgen_from_curves(boundary: LabelBoundary, h: float) -> _NetgenArrays:
    """Triangulate the box partitioned by the labelled 2D ``boundary`` (serial): every segment becomes a
    Netgen spline segment between shared points — so the curves meet conformingly at junctions and box
    edges — with ``leftdomain`` / ``rightdomain`` the regions on its two sides (a segment's left normal
    points into ``pairs[:, 1]``). Returns points, triangles and each triangle's region tag."""

    domains = _domains(boundary)

    def domain(side: int) -> int:
        return domains[side] if side >= 0 else 0

    geo = SplineGeometry()
    point_ids = [geo.AppendPoint(float(x), float(y)) for x, y in boundary.points]
    for (p, q), (right, left) in zip(boundary.elements.tolist(), boundary.pairs.tolist(), strict=True):
        geo.Append(["line", point_ids[p], point_ids[q]], leftdomain=domain(left), rightdomain=domain(right))
    ngmesh = geo.GenerateMesh(maxh=float(h))
    if not len(ngmesh.Elements2D()):
        raise RealizationError(f"Netgen could not triangulate the image geometry's curves at h = {h:g}; try another h")

    region_of_domain = np.zeros(len(domains) + 1, dtype=np.int32)
    for region, number in domains.items():
        region_of_domain[number] = region + 1
    points = np.array([list(p.p)[:2] for p in ngmesh.Points()], dtype=np.float64)
    cells = np.array([[v.nr - 1 for v in el.vertices] for el in ngmesh.Elements2D()], dtype=np.int64)
    material = region_of_domain[np.array([el.index for el in ngmesh.Elements2D()], dtype=np.int64)]
    return points, cells, material


def _netgen_from_surfaces(boundary: LabelBoundary, h: float) -> _NetgenArrays:
    """Tetrahedralize the box partitioned by the labelled 3D ``boundary`` (serial). Its triangles *are*
    the surface mesh — one closed, conforming (non-manifold at junction curves) surface — loaded in bulk
    under one ``FaceDescriptor`` per region pair (``domin`` / ``domout`` the regions the normal points
    from / into, 0 outside the box), then ``GenerateVolumeMesh`` fills each domain. Returns points,
    tetrahedra and each tetrahedron's region tag."""

    domains = _domains(boundary)

    def domain(side: int) -> int:
        return domains[side] if side >= 0 else 0

    mesh = NetgenMesh(dim=3)
    mesh.AddPoints(np.ascontiguousarray(boundary.points, dtype=np.float64))  # type: ignore[attr-defined]  # stubs lag
    pairs = boundary.pairs
    for k, (a, b) in enumerate(np.unique(pairs, axis=0).tolist()):
        chosen = (pairs[:, 0] == a) & (pairs[:, 1] == b)
        fd = mesh.Add(FaceDescriptor(surfnr=k + 1, domin=domain(a), domout=domain(b), bc=k + 1))
        triangles = np.ascontiguousarray(boundary.elements[chosen], dtype=np.int32)
        mesh.AddElements(dim=2, index=fd, data=triangles, base=0)  # type: ignore[attr-defined]
    mesh.GenerateVolumeMesh(maxh=float(h))  # type: ignore[call-arg]
    if not len(mesh.Elements3D()):
        # Netgen reports a failed domain on stdout and returns an empty mesh rather than raising
        raise RealizationError(
            f"Netgen could not tetrahedralize the image geometry's surfaces at h = {h:g} "
            "(an overlapping or self-intersecting boundary); try another h"
        )

    region_of_domain = np.zeros(len(domains) + 1, dtype=np.int32)
    for region, number in domains.items():
        region_of_domain[number] = region + 1
    points = np.asarray(mesh.Coordinates(), dtype=np.float64)  # type: ignore[attr-defined]
    elements = mesh.Elements3D().NumPy()  # type: ignore[attr-defined]
    cells = np.asarray(elements["nodes"], dtype=np.int64)[:, :4] - 1
    material = region_of_domain[np.asarray(elements["index"], dtype=np.int64)]
    return points, cells, material


def _facets_of_region(
    parent: dmesh.Mesh, cell_tags: dmesh.MeshTags, facets: NDArray[np.int32], region_tag: int
) -> NDArray[np.int32]:
    """The exterior ``facets`` whose (single) cell belongs to ``region_tag``."""

    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    cell_map = parent.topology.index_map(tdim)
    lookup = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    lookup[cell_tags.indices] = cell_tags.values
    offsets = np.asarray(facet_to_cell.offsets)
    cells = np.asarray(facet_to_cell.array)[offsets[np.asarray(facets, dtype=np.int64)]]
    selected: NDArray[np.int32] = np.asarray(facets, dtype=np.int32)[lookup[cells] == region_tag]
    return selected
