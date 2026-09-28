"""Geometry adapter, name→object loader, and the geometry cross-check.

A MathDescription references a geometry by *name* and declares only the
subdomain-class vocabulary it uses (§1.2.1); the concrete mesh and the
region→class assignment live here. For the v1 backend a `Geometry` maps each
subdomain-class name to a DOLFINx mesh and its kind. The `make_*_geometry` helpers
build bundled demo/test geometries gmsh-free — disks via the LGPL Netgen region
mesher, coupled two-compartment cells via `realize_interface_coupled` (see
LICENSING.md); the loader is a small name registry (one of the strategies §3.4
sanctions).

`cross_validate` is the part of §1.11.10 the formalism validator deferred because
it needs a Geometry: every subdomain class the MathDescription declares must
resolve to a region of matching kind, and the referenced geometry name must match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from dolfinx.mesh import (
    EntityMap,
    Mesh,
    MeshTags,
    create_rectangle,
    create_submesh,
    entities_to_geometry,
    exterior_facet_indices,
    locate_entities_boundary,
    refine,
)
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import MathDescription, SubdomainKind
from vcell_fenics.formalism.validator import Diagnostic


@dataclass(frozen=True)
class SubdomainGeometry:
    """The concrete mesh backing one subdomain class, plus its kind (for the
    cross-check)."""

    mesh: Mesh
    kind: SubdomainKind


@dataclass(frozen=True)
class BoundaryGeometry:
    """A labelled codim-1 boundary. `subdomains` lists the incident subdomain
    class(es): one name for an *external* boundary, two for an *internal* interface
    between two compartments. `facets` indexes the parent mesh's facets — for a
    single-compartment geometry the parent *is* the subdomain mesh, so the facets
    are equally on it. A boundary condition references the boundary by label."""

    subdomains: tuple[str, ...]
    facets: NDArray[np.int32]
    # a box face: on the domain's exterior however many subvolumes reach it (a face both the cytosol and the
    # extracellular space touch is still a `ds` boundary, not an interface between them)
    exterior: bool = False

    @property
    def is_internal(self) -> bool:
        """An internal interface is incident to two compartments (`dS`-integrable);
        an external boundary — a box face, or a boundary of one compartment — is `ds`-integrable."""

        return not self.exterior and len(self.subdomains) == 2


@dataclass(frozen=True)
class Geometry:
    """A named geometry: subdomain-class name → its mesh and kind, plus optional
    labelled boundaries for boundary conditions.

    For a *multi-compartment* geometry the subdomain meshes are submeshes of a shared
    `parent_mesh`; `cell_tags` marks each cell's compartment and `facet_tags` the
    labelled curves. These let the (future) mixed-dimensional assembly relate the
    compartments across an internal interface; for a single-compartment geometry they
    are `None` and each subdomain mesh stands alone."""

    name: str
    subdomains: dict[str, SubdomainGeometry]
    boundaries: dict[str, BoundaryGeometry] = field(default_factory=dict)
    parent_mesh: Mesh | None = None
    cell_tags: MeshTags | None = None
    facet_tags: MeshTags | None = None

    def kind_of(self, subdomain: str) -> SubdomainKind | None:
        entry = self.subdomains.get(subdomain)
        return entry.kind if entry is not None else None

    def mesh_of(self, subdomain: str) -> Mesh:
        return self.subdomains[subdomain].mesh

    def boundary_of(self, boundary: str) -> BoundaryGeometry | None:
        return self.boundaries.get(boundary)


# ---------------------------------------------------------------------------
# Name → object loader (a registry; §3.4 lists this as one valid strategy).
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Geometry] = {}


def register_geometry(geometry: Geometry) -> None:
    _REGISTRY[geometry.name] = geometry


def load_geometry(name: str) -> Geometry:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no geometry registered under {name!r}") from None


def clear_geometries() -> None:
    _REGISTRY.clear()


def _circle_disk_mesh(radius: float, h: float) -> Mesh:
    """A 2D triangle mesh of a disk of `radius` at resolution `h`, meshed gmsh-free by handing a fine
    circular boundary polyline to the LGPL Netgen region mesher (`mesh_region_netgen`). The boundary is
    meshed **directly** (nodes on the circle; area πR² to ~1e-4), unlike `realize` which extracts the disk
    as a submesh of a box partition — a direct-circle mesh matches the moving-boundary/ALE/slip solvers'
    node layout, which those tests are calibrated to. Netgen gives a quality isotropic interior.

    The circle polyline is CCW (linspace 0→2π of cos/sin) — the orientation `mesh_region_netgen` requires
    (it meshes the region to the left of each directed segment; a CW loop would stall on the exterior)."""

    from vcell_fenics.core.region_remesh_netgen import mesh_region_netgen  # lazy: loads Netgen, caps its pool

    npts = max(128, math.ceil(2.0 * math.pi * radius / h))
    theta = np.linspace(0.0, 2.0 * math.pi, npts, endpoint=False)
    loop = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return mesh_region_netgen(loop, h)


def make_disk_geometry(
    name: str, *, volume_subdomain: str, boundary: str | None = None, radius: float = 1.0, h: float = 0.1
) -> Geometry:
    """A bundled 2D disk exposed as a single `volume` subdomain class. When `boundary` is given, the
    disk's outer circle is registered as a labelled boundary under that name, so boundary conditions can
    target it.

    Meshed gmsh-free via the Netgen region mesher over a circular boundary polyline (`_circle_disk_mesh`)
    — a direct-circle disk (near-exact circle; quality interior). The boundary, when requested, is the
    mesh's exterior facets — the circle."""

    disk = _circle_disk_mesh(radius, h)
    subdomains = {volume_subdomain: SubdomainGeometry(mesh=disk, kind="volume")}
    boundaries: dict[str, BoundaryGeometry] = {}
    if boundary is not None:
        tdim = disk.topology.dim
        disk.topology.create_connectivity(tdim - 1, tdim)
        boundaries[boundary] = BoundaryGeometry(
            subdomains=(volume_subdomain,), facets=exterior_facet_indices(disk.topology)
        )
    return Geometry(name=name, subdomains=subdomains, boundaries=boundaries)


def make_structured_box_geometry(
    name: str,
    *,
    volume_subdomain: str,
    boundary: str | None = None,
    extent: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    h: float = 0.1,
) -> Geometry:
    """A 2D rectangle `[origin, origin + extent]` as a single `volume` subdomain, on a **structured**
    triangulated grid of `round(extent / h)` cells per side. **Nestable** (halving `h` doubles the grid and
    bisects every cell) *and* straight-boundary — so it carries **no** curved-boundary approximation error and
    a smooth P1 solution converges at the full **O(h²)**. The rigorous spatial-order geometry: pair it with
    `make_nested_disk_geometry` (curved, but P1-capped at ~O(h^1.5)) to separate a genuine order regression
    from the curved-domain variational crime. When `boundary` is given, the rectangle's exterior facets are
    registered under that name for boundary conditions."""

    nx = max(1, round(extent[0] / h))
    ny = max(1, round(extent[1] / h))
    lower = (origin[0], origin[1])
    upper = (origin[0] + extent[0], origin[1] + extent[1])
    mesh = create_rectangle(MPI.COMM_WORLD, [lower, upper], [nx, ny])
    subdomains = {volume_subdomain: SubdomainGeometry(mesh=mesh, kind="volume")}
    boundaries: dict[str, BoundaryGeometry] = {}
    if boundary is not None:
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        boundaries[boundary] = BoundaryGeometry(
            subdomains=(volume_subdomain,), facets=exterior_facet_indices(mesh.topology)
        )
    return Geometry(name=name, subdomains=subdomains, boundaries=boundaries)


def _snap_boundary_to_circle(mesh: Mesh, radius: float) -> None:
    """Move every boundary geometry node radially onto the circle of `radius` (in place). After a uniform
    `refine`, the new boundary nodes sit on the parent polygon's straight chords (radius < R); snapping them
    out restores an O(h²)-accurate curved boundary at the finer level."""
    boundary_vertices = locate_entities_boundary(mesh, 0, lambda x: np.full(x.shape[1], True, dtype=bool))
    nodes = np.unique(entities_to_geometry(mesh, 0, boundary_vertices).reshape(-1))
    coords = mesh.geometry.x
    r = np.hypot(coords[nodes, 0], coords[nodes, 1])
    r[r == 0.0] = 1.0  # a node exactly at the centre is never on the boundary; guard the divide
    coords[nodes, 0] *= radius / r
    coords[nodes, 1] *= radius / r


def make_nested_disk_geometry(
    name: str,
    *,
    volume_subdomain: str,
    boundary: str | None = None,
    radius: float = 1.0,
    h: float = 0.1,
    base_h: float = 0.1,
) -> Geometry:
    """A 2D disk (single `volume` subdomain) built by **uniformly refining** one fixed coarse Netgen mesh —
    `k = round(log2(base_h / h))` red refinements — and snapping the new boundary nodes onto the circle at
    each level. Unlike `make_disk_geometry`, whose Netgen mesh is rebuilt independently at each `h`, this is
    **nestable** *and* keeps the **curved boundary**: refinement bisects every cell so the coarse nodes are a
    strict subset of the fine (no mesh-topology noise), while the boundary stays O(h²) from the true circle
    (curvature preserved — the straight-edged box loses it). That makes it the right domain for a *spatial*
    convergence study on a curved geometry: an independently re-meshed disk floors the P1 order on remeshing
    noise, which the time-error-free MOL path reads directly. `h` must be `base_h / 2^k` (the runner uses
    `resolutions_h = [base_h, base_h/2, base_h/4, …]`). When `boundary` is given, the disk's exterior facets
    (the circle) are registered under that name for boundary conditions."""

    levels = max(0, round(math.log2(base_h / h)))
    disk = _circle_disk_mesh(radius, base_h)
    _snap_boundary_to_circle(disk, radius)
    for _ in range(levels):
        disk.topology.create_entities(1)  # refine bisects edges ⇒ they must exist first
        disk = refine(disk)[0]  # edges=None ⇒ uniform (red) refinement: nested
        _snap_boundary_to_circle(disk, radius)
    subdomains = {volume_subdomain: SubdomainGeometry(mesh=disk, kind="volume")}
    boundaries: dict[str, BoundaryGeometry] = {}
    if boundary is not None:
        tdim = disk.topology.dim
        disk.topology.create_connectivity(tdim - 1, tdim)
        boundaries[boundary] = BoundaryGeometry(
            subdomains=(volume_subdomain,), facets=exterior_facet_indices(disk.topology)
        )
    return Geometry(name=name, subdomains=subdomains, boundaries=boundaries)


def make_disk_membrane_geometry(name: str, *, surface_subdomain: str, radius: float = 1.0, h: float = 0.1) -> Geometry:
    """A bundled 2D disk's boundary, exposed as a single `surface` (codim-1) subdomain class — the
    membrane submesh.

    Meshed gmsh-free via the Netgen region mesher: the disk is meshed over a circular boundary polyline
    (`_circle_disk_mesh`) and its exterior facets (the near-exact circle) are extracted as the codim-1
    membrane submesh."""

    disk = _circle_disk_mesh(radius, h)
    tdim = disk.topology.dim
    disk.topology.create_connectivity(tdim - 1, tdim)
    membrane, *_ = create_submesh(disk, tdim - 1, exterior_facet_indices(disk.topology))
    return Geometry(name=name, subdomains={surface_subdomain: SubdomainGeometry(mesh=membrane, kind="surface")})


@dataclass(frozen=True)
class CoupledGeometry:
    """A bulk-surface coupled geometry for cross-mesh assembly: a `bulk` volume
    subdomain and a `surface` subdomain that *is* the bulk's boundary, related by an
    `EntityMap` so a form on the bulk's interface facets can reference surface
    functions (the §1.6.6 substrate). `interface` is the boundary label where bulk
    and surface couple (a facet tag on the bulk); `outer` is the external boundary
    (e.g. a reservoir). Distinct from `Geometry`: it carries the entity map and the
    bulk facet tags the mixed-dimensional assembler needs."""

    name: str
    bulk_subdomain: str
    surface_subdomain: str
    bulk_mesh: Mesh
    surface_mesh: Mesh
    entity_map: EntityMap
    facet_tags: MeshTags
    interface: str
    interface_tag: int
    outer: str
    outer_tag: int

    def kind_of(self, subdomain: str) -> SubdomainKind | None:
        if subdomain == self.bulk_subdomain:
            return "volume"
        if subdomain == self.surface_subdomain:
            return "surface"
        return None

    def mesh_of(self, subdomain: str) -> Mesh:
        if subdomain == self.bulk_subdomain:
            return self.bulk_mesh
        if subdomain == self.surface_subdomain:
            return self.surface_mesh
        raise KeyError(f"{subdomain!r} is not a subdomain of coupled geometry {self.name!r}")


# ---------------------------------------------------------------------------
# Two-bulk + membrane coupled geometry (§1.6.2 / §1.6.6, two-sided traces).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterfaceCoupledGeometry:
    """A coupled geometry for **two volume compartments meeting at a membrane** — the substrate
    for cross-compartment coupling where a term needs the traces of *both* bulk variables at the
    interface (a membrane equation in `trace(u_inner)` and `trace(u_outer)`, or one side's interface
    flux referencing the adjacent compartment's trace; §1.6.2 / §1.6.6).

    Unlike `CoupledGeometry` (one bulk + one surface-on-its-boundary, a single entity map), this
    carries **both** bulk submeshes, the membrane submesh, and the three `EntityMap`s relating them to
    `parent_mesh`. The membrane is an *interior* facet of the parent (a cell on each side), so a
    cross-mesh coupling form is integrated on the parent's interface facets via `dS` (not the bulk's
    `ds`), with all three submesh functions pulled in through their maps. The orientation-robust trace
    of a bulk variable on the membrane is `membrane_trace(f)` = `f('+') + f('-')` — each bulk function
    is non-zero only on its own side of the interface, so the sum picks its value regardless of which
    side the arbitrary `dS` restriction labels `+`."""

    name: str
    inner_subdomain: str
    outer_subdomain: str
    membrane_subdomain: str
    inner_mesh: Mesh
    outer_mesh: Mesh
    membrane_mesh: Mesh
    inner_entity_map: EntityMap
    outer_entity_map: EntityMap
    membrane_entity_map: EntityMap
    parent_mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags
    inner_region_tag: int  # the inner compartment's cell tag on the parent (for dx(parent)(region))
    outer_region_tag: int  # the outer compartment's cell tag on the parent
    interface: str
    interface_tag: int
    outer: str
    outer_tag: int

    def region_tag_of(self, subdomain: str) -> int:
        if subdomain == self.inner_subdomain:
            return self.inner_region_tag
        if subdomain == self.outer_subdomain:
            return self.outer_region_tag
        raise KeyError(f"{subdomain!r} is not a volume compartment of {self.name!r}")

    def kind_of(self, subdomain: str) -> SubdomainKind | None:
        if subdomain in (self.inner_subdomain, self.outer_subdomain):
            return "volume"
        if subdomain == self.membrane_subdomain:
            return "surface"
        return None

    def mesh_of(self, subdomain: str) -> Mesh:
        meshes = {
            self.inner_subdomain: self.inner_mesh,
            self.outer_subdomain: self.outer_mesh,
            self.membrane_subdomain: self.membrane_mesh,
        }
        try:
            return meshes[subdomain]
        except KeyError:
            raise KeyError(f"{subdomain!r} is not a subdomain of coupled geometry {self.name!r}") from None

    def entity_map_of(self, subdomain: str) -> EntityMap:
        maps = {
            self.inner_subdomain: self.inner_entity_map,
            self.outer_subdomain: self.outer_entity_map,
            self.membrane_subdomain: self.membrane_entity_map,
        }
        return maps[subdomain]


def membrane_trace(bulk_function: UflExpr) -> UflExpr:
    """The trace of a bulk variable on the interior membrane, written orientation-robustly for the
    parent `dS` measure: `bulk_function('+') + bulk_function('-')`. A bulk function is non-zero only on
    its own side of the interface (its entity map covers only that side's parent cells), so exactly one
    restriction is its real value and the other is zero — the sum is the trace regardless of which side
    `dS` happens to label `+`."""

    return bulk_function("+") + bulk_function("-")


def make_two_bulk_membrane_geometry(
    name: str,
    *,
    inner: str,
    outer_subdomain: str,
    membrane: str,
    interface: str,
    outer: str,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    h: float = 0.1,
) -> InterfaceCoupledGeometry:
    """A concentric two-compartment cell as an `InterfaceCoupledGeometry`: an inner-disk `inner`
    compartment of radius `inner_radius` inside an **annular** `outer_subdomain` of outer radius
    `outer_radius`, meeting at the `membrane`. `interface` labels the shared membrane circle; `outer`
    labels the reservoir **wall** — the annulus's outer circle — where a coupling model may hold a
    Dirichlet. Carries the three entity maps so a coupling form can reference both bulk traces at the
    membrane.

    Realized gmsh-free via `realize_interface_coupled`'s `background_subdomain` path: a 3-region
    body-fitted description (inner disk / annular outer / discarded background) is realized and then
    bounded to inner + outer, so the annulus's outer circle is a true exterior reservoir wall — the same
    disk-in-annulus geometry the old gmsh mesher produced (area π(R_out²−R_in²) for the annulus, a
    circular `ds` wall), without gmsh."""

    from vcell_fenics.backend.realize import realize_interface_coupled  # lazy: realize imports this module

    # Body-fit both circles inside a box large enough that the background surrounds the annulus (so its
    # outer circle is interior to the box and becomes the bounded parent's exterior wall). The two
    # analytic shapes are nested disks; priority (declaration order, inner first) assigns each cell.
    half = 1.3 * outer_radius
    r_in2, r_out2 = inner_radius * inner_radius, outer_radius * outer_radius
    background = "_background"
    desc = GeometryDescription(
        name=name,
        dim=2,
        extent=(2.0 * half, 2.0 * half, 1.0),
        origin=(-half, -half, 0.0),
        subvolumes=(
            SubVolume(name=inner, type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {r_in2}"),
            SubVolume(name=outer_subdomain, type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {r_out2}"),
            SubVolume(name=background, type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name=membrane, inside=inner, outside=outer_subdomain),),
    )
    return realize_interface_coupled(
        desc,
        inner_subdomain=inner,
        outer_subdomain=outer_subdomain,
        membrane_subdomain=membrane,
        interface=interface,
        outer=outer,
        background_subdomain=background,
        h=h,
    )


# ---------------------------------------------------------------------------
# §1.11.10 geometry cross-check (deferred from the formalism validator).
# ---------------------------------------------------------------------------


def cross_validate(md: MathDescription, geometry: Geometry) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if md.geometry != geometry.name:
        diagnostics.append(
            Diagnostic(
                "error",
                "geometry",
                f"MathDescription references geometry {md.geometry!r}, but {geometry.name!r} was provided (§3.4)",
            )
        )
    for i, subdomain in enumerate(md.subdomains):
        geom_kind = geometry.kind_of(subdomain.name)
        if geom_kind is None:
            diagnostics.append(
                Diagnostic(
                    "error",
                    f"subdomains[{i}]",
                    f"subdomain class {subdomain.name!r} has no matching region in geometry "
                    f"{geometry.name!r} (§1.11.10)",
                )
            )
        elif geom_kind != subdomain.kind:
            diagnostics.append(
                Diagnostic(
                    "error",
                    f"subdomains[{i}]",
                    f"subdomain {subdomain.name!r} is declared kind {subdomain.kind!r} but the geometry "
                    f"provides {geom_kind!r} (§1.11.10)",
                )
            )
    # Every BC's labelled boundary must resolve in the geometry — the §1.11.10 check
    # the formalism validator deferred because a boundary is a geometry-side entity.
    for i, bc in enumerate(md.boundary_conditions):
        if geometry.boundary_of(bc.boundary) is None:
            diagnostics.append(
                Diagnostic(
                    "error",
                    f"boundary_conditions[{i}]",
                    f"boundary {bc.boundary!r} has no matching labelled boundary in geometry "
                    f"{geometry.name!r} (§1.11.10)",
                )
            )
    return diagnostics
