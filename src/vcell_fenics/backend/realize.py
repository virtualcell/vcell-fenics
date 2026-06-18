"""Realize a :class:`~vcell_fenics.formalism.geometry_schema.GeometryDescription` into a concrete
backend :class:`~vcell_fenics.backend.geometry.Geometry` (ADR 007; `geometric-formalism.md` §3).

This is the bridge from the declarative geometry *spec* to a meshed object the backend solves on —
the realization layer §3.1 describes. It subsumes the imperative ``make_*`` helpers as recipes over
the formalism.

**Realization v1:**

- **Trivial / non-spatial (``dim = 0``).** A well-mixed geometry: each ``compartmental`` subvolume
  (and any membrane between them) is backed by a minimal single-cell mesh the lumped-ODE templates
  carry a constant over — a representational stand-in, not a spatial domain.
- **2D body-fitted (``dim = 2``), box-partitioned.** The geometry is the bounding box (``extent`` /
  ``origin``) partitioned by analytic subvolumes; the outer boundary is the **box faces**
  (``x_minus`` / ``x_plus`` / ``y_minus`` / ``y_plus``) and the membrane is the internal interface.
  Each analytic subvolume's boolean predicate is lowered to a Rvachev implicit field
  (`formalism/rvachev.py`), sampled on a grid, and its ``φ = 0`` contour is **marched** (scikit-image)
  and embedded in a gmsh model that fragments the box into the regions. v1 supports the common
  *one interior subvolume + background* topology (one closed contour, strictly inside the box).

**Not yet here:** 3D, multi-region (>2 subvolumes) partitions, ``image`` meshing, and the unfitted
(level-set / cut-FEM) consumption of the same field. Unsupported descriptions raise
:class:`NotImplementedError`.
"""

from __future__ import annotations

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
from vcell_fenics.formalism.rvachev import subvolume_implicit_functions

_FACE_NAMES_2D = ("x_minus", "x_plus", "y_minus", "y_plus")
# Physical-group tag layout for the 2D realization.
_MEMBRANE_TAG = 100
_FACE_TAG_BASE = 200  # x_minus=200, x_plus=201, …


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
    if len(subvolumes) != 2:
        raise NotImplementedError(
            f"2D realization v1 supports exactly one interior subvolume + background; "
            f"geometry {description.name!r} has {len(subvolumes)} subvolumes"
        )
    interior, background = subvolumes[0], subvolumes[1]
    if interior.type != "analytic":
        raise RealizationError(
            f"2D geometry {description.name!r} interior subvolume {interior.name!r} must be 'analytic', "
            f"got {interior.type!r}"
        )

    ox, oy = description.origin[0], description.origin[1]
    lx, ly = description.extent[0], description.extent[1]
    field = subvolume_implicit_functions(description)[interior.name]
    contour = _march_interior_contour(field, ox=ox, oy=oy, lx=lx, ly=ly, resolution=resolution, name=description.name)

    membrane_name = description.surfaces[0].name if description.surfaces else None
    parent = _mesh_box_with_contour(contour, ox=ox, oy=oy, lx=lx, ly=ly, h=h, comm=comm, name=description.name)
    cell_tags, facet_tags, region_tags = _classify_and_tag(
        parent, field, names=(interior.name, background.name), ox=ox, oy=oy, lx=lx, ly=ly
    )

    tdim = parent.topology.dim
    subdomains: dict[str, SubdomainGeometry] = {}
    for name, tag in region_tags.items():
        sub_mesh, *_ = dmesh.create_submesh(parent, tdim, cell_tags.find(tag))
        subdomains[name] = SubdomainGeometry(mesh=sub_mesh, kind="volume")

    boundaries: dict[str, BoundaryGeometry] = {}
    if membrane_name is not None:
        membrane_mesh, *_ = dmesh.create_submesh(parent, tdim - 1, facet_tags.find(_MEMBRANE_TAG))
        subdomains[membrane_name] = SubdomainGeometry(mesh=membrane_mesh, kind="surface")
        boundaries[membrane_name] = BoundaryGeometry(
            subdomains=(interior.name, background.name), facets=facet_tags.find(_MEMBRANE_TAG)
        )
    # The box faces bound the region that touches them — the background.
    for i, face in enumerate(_FACE_NAMES_2D):
        facets = facet_tags.find(_FACE_TAG_BASE + i)
        if facets.size:
            boundaries[face] = BoundaryGeometry(subdomains=(background.name,), facets=facets)

    return Geometry(
        name=description.name,
        subdomains=subdomains,
        boundaries=boundaries,
        parent_mesh=parent,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
    )


def _march_interior_contour(
    field: Expr, *, ox: float, oy: float, lx: float, ly: float, resolution: int, name: str
) -> NDArray[np.float64]:
    """Sample ``field`` over the box and march its ``φ = 0`` contour, returned as physical (x, y)
    vertices. v1 requires a single closed contour strictly inside the box (the membrane)."""

    xs = np.linspace(ox, ox + lx, resolution)
    ys = np.linspace(oy, oy + ly, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    phi = _eval_field(field, (grid_x, grid_y))

    contours = find_contours(phi, 0.0)
    if len(contours) != 1:
        raise NotImplementedError(
            f"2D realization v1 expects one interior membrane contour for {name!r}, found {len(contours)} "
            "(nested / multi-region geometries are a later slice)"
        )
    rows_cols = contours[0]
    dx, dy = lx / (resolution - 1), ly / (resolution - 1)
    xy = np.column_stack([ox + rows_cols[:, 1] * dx, oy + rows_cols[:, 0] * dy])
    on_edge = (
        np.isclose(xy[:, 0], ox)
        | np.isclose(xy[:, 0], ox + lx)
        | np.isclose(xy[:, 1], oy)
        | np.isclose(xy[:, 1], oy + ly)
    )
    if on_edge.any():
        raise NotImplementedError(
            f"2D realization v1 requires the membrane of {name!r} to be strictly inside the box "
            "(a subvolume touching the box boundary is a later slice)"
        )
    return xy


def _mesh_box_with_contour(
    contour: NDArray[np.float64], *, ox: float, oy: float, lx: float, ly: float, h: float, comm: MPI.Comm, name: str
) -> dmesh.Mesh:
    """Build a gmsh model of the box with ``contour`` embedded and fragment it so the contour is a
    conforming internal edge, returning the body-fitted DOLFINx mesh. Region / membrane / face
    tagging is done afterwards in DOLFINx (:func:`_classify_and_tag`) — robustly, by field sign —
    rather than from gmsh surface areas (which mis-assign when the interior is the larger region)."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add(name)
        occ = gmsh.model.occ
        # Simplify the marched polyline (Douglas–Peucker) to a fraction of the mesh size: removes
        # the sub-grid near-duplicate vertices OCC rejects, while staying within h of the contour.
        verts = approximate_polygon(contour, tolerance=0.25 * h)
        if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        if len(verts) < 3:
            raise RealizationError(f"membrane contour of {name!r} degenerated to {len(verts)} vertices")
        points = [occ.addPoint(float(x), float(y), 0.0, h) for x, y in verts]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        inner = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        box = occ.addRectangle(ox, oy, 0.0, lx, ly)
        occ.fragment([(2, box)], [(2, inner)])
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
    parent: dmesh.Mesh, field: Expr, *, names: tuple[str, str], ox: float, oy: float, lx: float, ly: float
) -> tuple[dmesh.MeshTags, dmesh.MeshTags, dict[str, int]]:
    """Tag the body-fitted mesh by the implicit field's sign — each cell is assigned to the region
    whose `φ` is negative at the cell midpoint (the per-cell analogue of SBML Spatial's interior
    points). Robust to which region is larger or non-convex, unlike an area or centroid heuristic.
    The membrane is the set of interior facets between cells of different regions; the box faces are
    the exterior facets, classified by position."""

    interior_name, background_name = names
    region_tags = {interior_name: 1, background_name: 2}
    tdim = parent.topology.dim

    cells = np.arange(parent.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_mid = dmesh.compute_midpoints(parent, tdim, cells)
    phi = _eval_field(field, (cell_mid[:, 0], cell_mid[:, 1]))
    cell_values = np.where(phi < 0, region_tags[interior_name], region_tags[background_name]).astype(np.int32)
    cell_tags = dmesh.meshtags(parent, tdim, cells, cell_values)

    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    num_facets = parent.topology.index_map(tdim - 1).size_local
    facet_mid = dmesh.compute_midpoints(parent, tdim - 1, np.arange(num_facets, dtype=np.int32))
    facet_index: list[int] = []
    facet_value: list[int] = []
    for facet in range(num_facets):
        incident = facet_to_cell.links(facet)
        if incident.size == 1:  # exterior — a box face
            face = _classify_face(float(facet_mid[facet, 0]), float(facet_mid[facet, 1]), ox=ox, oy=oy, lx=lx, ly=ly)
            if face is not None:
                facet_index.append(facet)
                facet_value.append(_FACE_TAG_BASE + face)
        elif cell_values[incident[0]] != cell_values[incident[1]]:  # interior region boundary — membrane
            facet_index.append(facet)
            facet_value.append(_MEMBRANE_TAG)

    order = np.argsort(facet_index)
    facet_tags = dmesh.meshtags(
        parent,
        tdim - 1,
        np.asarray(facet_index, dtype=np.int32)[order],
        np.asarray(facet_value, dtype=np.int32)[order],
    )
    return cell_tags, facet_tags, region_tags


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
