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
  grid, and its ``φ = 0`` boundary is **marched** (scikit-image) and embedded in a gmsh model that
  fragments the box into the regions (nested shapes nest; disjoint shapes sit side by side). Each
  cell is then assigned to the subvolume whose priority-resolved field is negative at its midpoint;
  membrane facets between two regions are named by the ``SurfaceClass`` for that pair.

**Not yet here:** 3D, ``image`` meshing, subvolumes touching the box boundary, and the unfitted
(level-set / cut-FEM) consumption of the same field. Unsupported descriptions raise
:class:`NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI
from numpy.typing import NDArray
from skimage.measure import approximate_polygon, find_contours

from vcell_fenics.backend.geometry import BoundaryGeometry, Geometry, SubdomainGeometry
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import lower_predicate, subvolume_implicit_functions

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
    resolution: int = 201,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Realize ``description`` into a backend :class:`Geometry`.

    ``h`` is the target mesh size and ``resolution`` the per-axis sampling grid for the implicit
    field (2D only). The result is *not* registered — the caller passes it to
    :func:`~vcell_fenics.backend.geometry.register_geometry` if name resolution is wanted.
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


def _realize_2d(description: GeometryDescription, *, h: float, resolution: int, comm: MPI.Comm) -> Geometry:
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

    # March each shape's own (raw, not priority-resolved) boundary so the mesh conforms to every
    # analytic surface; priority then decides each cell's owner. For nested shapes (nucleus in
    # cytosol in ecm) the contours nest; for disjoint shapes they sit side by side.
    contours: list[NDArray[np.float64]] = []
    for subvolume in subvolumes[:-1]:
        assert subvolume.expression is not None  # checked above
        raw_field = lower_predicate(parse(subvolume.expression))
        contours.extend(
            _march_contours(raw_field, ox=ox, oy=oy, lx=lx, ly=ly, resolution=resolution, name=subvolume.name)
        )
    if not contours:
        raise RealizationError(f"2D geometry {description.name!r} has no interior contours to mesh")

    parent = _mesh_box_with_contours(contours, ox=ox, oy=oy, lx=lx, ly=ly, h=h, comm=comm, name=description.name)
    fields = subvolume_implicit_functions(description)  # priority-resolved; classifies each cell
    tagging = _classify_and_tag(parent, description, fields, ox=ox, oy=oy, lx=lx, ly=ly)
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
    field: Expr, *, ox: float, oy: float, lx: float, ly: float, resolution: int, name: str
) -> list[NDArray[np.float64]]:
    """Sample ``field`` over the box and march every ``φ = 0`` contour, each returned as physical
    (x, y) vertices. v1 requires each contour to be strictly inside the box (a subvolume touching
    the box boundary is a later slice)."""

    xs = np.linspace(ox, ox + lx, resolution)
    ys = np.linspace(oy, oy + ly, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    phi = _eval_field(field, (grid_x, grid_y))

    dx, dy = lx / (resolution - 1), ly / (resolution - 1)
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
) -> dmesh.Mesh:
    """Build a gmsh model of the box with every ``contours`` polyline embedded and fragment it so
    each is a conforming internal edge, returning the body-fitted DOLFINx mesh. Region / membrane /
    face tagging is done afterwards in DOLFINx (:func:`_classify_and_tag`), by field sign."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add(name)
        occ = gmsh.model.occ
        tools: list[tuple[int, int]] = []
        for contour in contours:
            # Simplify the marched polyline (Douglas–Peucker) to a fraction of the mesh size: removes
            # the sub-grid near-duplicate vertices OCC rejects, while staying within h of the contour.
            verts = approximate_polygon(contour, tolerance=0.25 * h)
            if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
                verts = verts[:-1]
            if len(verts) < 3:
                raise RealizationError(f"a contour of {name!r} degenerated to {len(verts)} vertices")
            points = [occ.addPoint(float(x), float(y), 0.0, h) for x, y in verts]
            lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
            tools.append((2, occ.addPlaneSurface([occ.addCurveLoop(lines)])))
        box = occ.addRectangle(ox, oy, 0.0, lx, ly)
        occ.fragment([(2, box)], tools)
        occ.synchronize()
        # One physical group over every fragment so model_to_mesh returns the whole mesh.
        gmsh.model.addPhysicalGroup(2, [s for _, s in gmsh.model.getEntities(2)], tag=1, name="domain")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        mesh = model_to_mesh(gmsh.model, comm, rank=0, gdim=2).mesh
    finally:
        gmsh.finalize()
    return mesh


def _classify_and_tag(
    parent: dmesh.Mesh,
    description: GeometryDescription,
    fields: dict[str, Expr],
    *,
    ox: float,
    oy: float,
    lx: float,
    ly: float,
) -> _Tagging:
    """Tag the body-fitted mesh by the priority-resolved implicit fields — each cell is assigned to
    the subvolume whose `φ` is negative at the cell midpoint (the per-cell analogue of SBML Spatial's
    interior points; the fields partition the plane, so exactly one is negative). A membrane facet
    (interior, between two regions) is named by the `SurfaceClass` for that subvolume pair; box faces
    are exterior facets, classified by position and the region they touch."""

    subvolumes = description.subvolumes
    region_tags = {sv.name: i + 1 for i, sv in enumerate(subvolumes)}
    tag_to_name = {tag: name for name, tag in region_tags.items()}
    tdim = parent.topology.dim

    cells = np.arange(parent.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_mid = dmesh.compute_midpoints(parent, tdim, cells)
    cell_values = np.zeros(cells.size, dtype=np.int32)
    for name, tag in region_tags.items():
        inside = _eval_field(fields[name], (cell_mid[:, 0], cell_mid[:, 1])) < 0
        cell_values[inside] = tag
    cell_values[cell_values == 0] = region_tags[subvolumes[-1].name]  # any unclaimed cell → background
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
