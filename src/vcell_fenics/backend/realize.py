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

**Not yet here:** 3D, ``image`` meshing, subvolumes touching the box boundary, and the unfitted
(level-set / cut-FEM) consumption of the same field. Unsupported descriptions raise
:class:`NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import numpy as np
import pyngcore
import ufl
from dolfinx import mesh as dmesh
from mpi4py import MPI
from numpy.typing import NDArray
from skimage.measure import approximate_polygon, find_contours

from vcell_fenics.backend.geometry import (
    BoundaryGeometry,
    Geometry,
    InterfaceCoupledGeometry,
    SubdomainGeometry,
)
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import lower_predicate, subvolume_implicit_functions

# Netgen's default multi-threaded TaskManager busy-waits in a long-lived process (ADR 008 §3); the
# realization meshes are small and serial, so cap the pool before the mesher loads. Its import must
# follow the cap, so it sits after the other imports (E402) rather than in the sorted block above.
pyngcore.SetNumThreads(1)

from netgen.geom2d import SplineGeometry  # noqa: E402  (must follow SetNumThreads)

_FACE_NAMES_2D = ("x_minus", "x_plus", "y_minus", "y_plus")
# Mesh-tag layout for the 2D realization: one tag per declared surface class (membrane), one per box
# face. Region (cell) tags are 1..N over the subvolumes.
_SURFACE_TAG_BASE = 100  # membrane tags, in SurfaceClass order
_FACE_TAG_BASE = 200  # x_minus=200, x_plus=201, …


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


class RealizationError(ValueError):
    """A :class:`GeometryDescription` is structurally well-formed but cannot be realized into a
    backend :class:`Geometry` (e.g. a non-spatial geometry carrying a spatial subvolume type, or a
    spatial topology v1 does not yet mesh)."""


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
    raise NotImplementedError(
        f"realization of dim={description.dim} geometry {description.name!r} is not implemented yet "
        "(v1 supports dim 0 and dim 2)"
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
    tagging = _classify_and_tag(parent, description, fields, cell_face=cell_face, ox=ox, oy=oy, lx=lx, ly=ly)
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
    tdim = parent.topology.dim

    subdomains: dict[str, SubdomainGeometry] = {}
    for name, tag in tagging.region_tags.items():
        sub_mesh, *_ = dmesh.create_submesh(parent, tdim, tagging.cell_tags.find(tag))
        subdomains[name] = SubdomainGeometry(mesh=sub_mesh, kind="volume")

    boundaries: dict[str, BoundaryGeometry] = {}
    for surface in description.surfaces:
        facets = tagging.facet_tags.find(tagging.surface_tags[surface.name])
        if not facets.size:
            continue  # the declared membrane has no realized interface (e.g. regions don't meet)
        membrane_mesh, *_ = dmesh.create_submesh(parent, tdim - 1, facets)
        subdomains[surface.name] = SubdomainGeometry(mesh=membrane_mesh, kind="surface")
        boundaries[surface.name] = BoundaryGeometry(subdomains=(surface.inside, surface.outside), facets=facets)

    for i, face in enumerate(_FACE_NAMES_2D):
        facets = tagging.facet_tags.find(_FACE_TAG_BASE + i)
        if facets.size:
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


def realize_interface_coupled(
    description: GeometryDescription,
    *,
    inner_subdomain: str,
    outer_subdomain: str,
    membrane_subdomain: str,
    interface: str,
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
    """

    expected = {inner_subdomain, outer_subdomain}
    volume_names = {sv.name for sv in description.subvolumes}
    if not expected <= volume_names:
        raise RealizationError(
            f"interface-coupled realization needs both compartments {sorted(expected)} among the "
            f"geometry's subvolumes {sorted(volume_names)}"
        )
    surface = next((s for s in description.surfaces if s.name == membrane_subdomain), None)
    if surface is None:
        raise RealizationError(
            f"interface-coupled realization needs a SurfaceClass named {membrane_subdomain!r}; "
            f"geometry {description.name!r} has {[s.name for s in description.surfaces]}"
        )

    parent, tagging = _realize_2d_partition(description, h=h, resolution=resolution, comm=comm)
    tdim = parent.topology.dim
    interface_facets = tagging.facet_tags.find(tagging.surface_tags[membrane_subdomain])
    if not interface_facets.size:
        raise RealizationError(f"the membrane {membrane_subdomain!r} has no realized interface facets")

    # Retain the entity maps (realize() discards them) — they relate each submesh to the parent so the
    # coupling form on the parent's interface dS can pull in both bulk traces.
    inner_mesh, inner_emap, *_ = dmesh.create_submesh(
        parent, tdim, tagging.cell_tags.find(tagging.region_tags[inner_subdomain])
    )
    outer_mesh, outer_emap, *_ = dmesh.create_submesh(
        parent, tdim, tagging.cell_tags.find(tagging.region_tags[outer_subdomain])
    )
    membrane_mesh, membrane_emap, *_ = dmesh.create_submesh(parent, tdim - 1, interface_facets)

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
        cell_tags=tagging.cell_tags,
        facet_tags=tagging.facet_tags,
        inner_region_tag=tagging.region_tags[inner_subdomain],
        outer_region_tag=tagging.region_tags[outer_subdomain],
        interface=interface,
        interface_tag=tagging.surface_tags[membrane_subdomain],
        # The external boundary is the box faces (no single reservoir circle); a reservoir Dirichlet
        # there is a follow-up, and `integrate_interface_coupled` does not use these fields.
        outer="exterior",
        outer_tag=-1,
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
    """Build a Netgen model of the box with every ``contours`` polyline embedded as a conforming
    internal boundary, returning the body-fitted DOLFINx mesh **and** a cell→region tag. Netgen assigns
    a material index per element (``el.index``) from the leftdomain/rightdomain of the embedded curves;
    that index is one owner per body-fit region — exactly the tag :func:`_classify_and_tag` needs to
    keep each region whole. The containment forest of the (non-intersecting) contours supplies those
    domains: a top-level contour separates its interior from the box background, a nested one from its
    parent's interior (nested shapes nest, disjoint shapes sit side by side). LGPL Netgen replaces GPL
    gmsh here (ADR 008); region / membrane / face tagging still happens afterwards in DOLFINx."""

    # Simplify (Douglas–Peucker, to a fraction of the mesh size) and orient each contour CCW so its
    # interior is on the left — Netgen's ``leftdomain``. Simplification drops the sub-grid near-duplicate
    # marching vertices while staying within h of the contour.
    polys: list[NDArray[np.float64]] = []
    for contour in contours:
        verts = approximate_polygon(contour, tolerance=0.25 * h)
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
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    mesh = dmesh.create_mesh(comm, cells, domain, points)

    # create_mesh may reorder cells for locality; realign the per-cell material via the input index.
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    local_cells = np.arange(n_local, dtype=np.int32)
    cell_material = material[np.asarray(mesh.topology.original_cell_index)[:n_local]]
    cell_face = dmesh.meshtags(mesh, tdim, local_cells, cell_material)
    return mesh, cell_face


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


def _classify_and_tag(
    parent: dmesh.Mesh,
    description: GeometryDescription,
    fields: dict[str, Expr],
    *,
    cell_face: dmesh.MeshTags,
    ox: float,
    oy: float,
    lx: float,
    ly: float,
) -> _Tagging:
    """Tag the body-fitted mesh by the priority-resolved implicit fields. Each cell is provisionally
    assigned to the subvolume whose `φ` is negative at its midpoint (the fields partition the plane, so
    exactly one is negative), then each fragment FACE is given its **majority** vote: a face is one
    body-fit region, so resolving it as a whole keeps a boundary cell — body-fit outside the simplified
    contour but with its midpoint still inside the exact shape — from being mis-assigned across the
    boundary (which would bulge the region boundary by ~½ a cell). A membrane facet (interior, between two
    regions) is named by the `SurfaceClass` for that pair; box faces are exterior facets, classified by
    position and the region they touch."""

    subvolumes = description.subvolumes
    region_tags = {sv.name: i + 1 for i, sv in enumerate(subvolumes)}
    tag_to_name = {tag: name for name, tag in region_tags.items()}
    tdim = parent.topology.dim

    cells = np.arange(parent.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_mid = dmesh.compute_midpoints(parent, tdim, cells)
    per_cell = np.zeros(cells.size, dtype=np.int32)
    for name, tag in region_tags.items():
        per_cell[_eval_field(fields[name], (cell_mid[:, 0], cell_mid[:, 1])) < 0] = tag
    per_cell[per_cell == 0] = region_tags[subvolumes[-1].name]  # any unclaimed cell → background

    # Resolve each body-fit face as a whole by the majority of its cells' votes (consistent with the
    # conforming boundary, vs a per-cell midpoint test that frays it).
    cell_values = per_cell.copy()
    face_of_cell = np.zeros(cells.size, dtype=cell_face.values.dtype)
    face_of_cell[cell_face.indices] = cell_face.values
    for face_id in np.unique(face_of_cell):
        in_face = face_of_cell == face_id
        cell_values[in_face] = np.bincount(per_cell[in_face]).argmax()
    cell_tags = dmesh.meshtags(parent, tdim, cells, cell_values)

    surface_tags = {sc.name: _SURFACE_TAG_BASE + i for i, sc in enumerate(description.surfaces)}
    pair_to_tag = {
        frozenset({sc.inside, sc.outside}): _SURFACE_TAG_BASE + i for i, sc in enumerate(description.surfaces)
    }

    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    num_facets = parent.topology.index_map(tdim - 1).size_local
    facet_mid = dmesh.compute_midpoints(parent, tdim - 1, np.arange(num_facets, dtype=np.int32))
    facet_index: list[int] = []
    facet_value: list[int] = []
    face_regions: dict[int, set[str]] = {}
    for facet in range(num_facets):
        incident = facet_to_cell.links(facet)
        if incident.size == 1:  # exterior — a box face
            face = _classify_face(float(facet_mid[facet, 0]), float(facet_mid[facet, 1]), ox=ox, oy=oy, lx=lx, ly=ly)
            if face is not None:
                tag = _FACE_TAG_BASE + face
                facet_index.append(facet)
                facet_value.append(tag)
                face_regions.setdefault(tag, set()).add(tag_to_name[int(cell_values[incident[0]])])
            continue
        left, right = int(cell_values[incident[0]]), int(cell_values[incident[1]])
        if left == right:
            continue  # interior to one region
        membrane_tag = pair_to_tag.get(frozenset({tag_to_name[left], tag_to_name[right]}))
        if membrane_tag is not None:  # a declared SurfaceClass names this interface
            facet_index.append(facet)
            facet_value.append(membrane_tag)

    order = np.argsort(facet_index)
    facet_tags = dmesh.meshtags(
        parent,
        tdim - 1,
        np.asarray(facet_index, dtype=np.int32)[order],
        np.asarray(facet_value, dtype=np.int32)[order],
    )
    return _Tagging(
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        region_tags=region_tags,
        surface_tags=surface_tags,
        face_regions={tag: tuple(sorted(names)) for tag, names in face_regions.items()},
    )


def _classify_face(x: float, y: float, *, ox: float, oy: float, lx: float, ly: float) -> int | None:
    """Index into ``_FACE_NAMES_2D`` for a facet centroid on a box edge, else ``None`` (membrane)."""

    if np.isclose(x, ox):
        return 0  # x_minus
    if np.isclose(x, ox + lx):
        return 1  # x_plus
    if np.isclose(y, oy):
        return 2  # y_minus
    if np.isclose(y, oy + ly):
        return 3  # y_plus
    return None


def _eval_field(expr: Expr, coords: tuple[NDArray[np.float64], NDArray[np.float64]]) -> NDArray[np.float64]:
    """Vectorised numeric evaluation of a (lowered, geom.x-only) implicit-function expression over a
    coordinate grid. Handles the arithmetic / min / max / elementary-function subset an implicit
    function contains; relational or unbound nodes (which a lowered field never has) raise."""

    if isinstance(expr, Number):
        return np.full_like(coords[0], expr.value, dtype=np.float64)
    if isinstance(expr, IndexAccess):
        return coords[int(_constant(expr.index))]
    if isinstance(expr, UnaryOp):
        value = _eval_field(expr.operand, coords)
        return -value if expr.op == "-" else value
    if isinstance(expr, BinaryOp):
        left, right = _eval_field(expr.left, coords), _eval_field(expr.right, coords)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return left**right
        raise RealizationError(f"operator {expr.op!r} is not valid in an implicit function")
    if isinstance(expr, FunctionCall):
        args = [_eval_field(a, coords) for a in expr.args]
        return _call(expr.callee, args)
    raise RealizationError(f"cannot evaluate {type(expr).__name__} in an implicit function (unbound name?)")


def _call(callee: str, args: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    if callee == "min":
        return np.minimum(args[0], args[1])
    if callee == "max":
        return np.maximum(args[0], args[1])
    unary = {"sqrt": np.sqrt, "abs": np.abs, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos}
    if callee in unary:
        return unary[callee](args[0])
    if callee == "pow":
        return np.power(args[0], args[1])
    raise RealizationError(f"function {callee!r} is not supported in an implicit function")


def _constant(expr: Expr) -> float:
    if isinstance(expr, Number):
        return expr.value
    raise RealizationError("expected a constant index in geom.x[…]")
