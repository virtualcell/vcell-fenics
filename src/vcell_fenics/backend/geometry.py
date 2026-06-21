"""Geometry adapter, name→object loader, and the geometry cross-check.

A MathDescription references a geometry by *name* and declares only the
subdomain-class vocabulary it uses (§1.2.1); the concrete mesh and the
region→class assignment live here. For the v1 backend a `Geometry` maps each
subdomain-class name to a DOLFINx mesh and its kind. `make_disk_geometry` wraps
the existing `approaches/static` disk; the loader is a small name registry (one
of the strategies §3.4 sanctions).

`cross_validate` is the part of §1.11.10 the formalism validator deferred because
it needs a Geometry: every subdomain class the MathDescription declares must
resolve to a region of matching kind, and the referenced geometry name must match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dolfinx.mesh import EntityMap, Mesh, MeshTags
from numpy.typing import NDArray

from vcell_fenics.approaches.multicompartment.geometry import (
    MEMBRANE_TAG,
    OUTER_TAG,
    create_cell_extracellular,
    create_extracellular_annulus,
)
from vcell_fenics.approaches.static.geometry import BOUNDARY_TAG, create_disk
from vcell_fenics.approaches.submesh.geometry import create_disk_with_membrane
from vcell_fenics.backend._typing import UflExpr
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

    @property
    def is_internal(self) -> bool:
        """An internal interface is incident to two compartments (`dS`-integrable);
        an external boundary to one (`ds`-integrable)."""

        return len(self.subdomains) == 2


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


def make_disk_geometry(
    name: str, *, volume_subdomain: str, boundary: str | None = None, radius: float = 1.0, h: float = 0.1
) -> Geometry:
    """A bundled 2D disk exposed as a single `volume` subdomain class. When
    `boundary` is given, the disk's outer circle is registered as a labelled
    boundary under that name, so boundary conditions can target it."""

    disk = create_disk(radius=radius, h=h)
    subdomains = {volume_subdomain: SubdomainGeometry(mesh=disk.mesh, kind="volume")}
    boundaries: dict[str, BoundaryGeometry] = {}
    if boundary is not None:
        facets = disk.facet_tags.find(BOUNDARY_TAG)
        boundaries[boundary] = BoundaryGeometry(subdomains=(volume_subdomain,), facets=facets)
    return Geometry(name=name, subdomains=subdomains, boundaries=boundaries)


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
    """A concentric two-compartment cell: an inner-disk `cytosol` and outer-annulus
    `extracellular` (both `volume`), plus the `membrane` between them as a `surface`
    subdomain. The membrane is also registered as the internal boundary `interface`
    (incident to both compartments); the outer circle is the external boundary
    `outer` (incident to the extracellular space only)."""

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


def make_disk_membrane_geometry(name: str, *, surface_subdomain: str, radius: float = 1.0, h: float = 0.1) -> Geometry:
    """A bundled 2D disk's boundary, exposed as a single `surface` (codim-1)
    subdomain class — the membrane submesh."""

    submesh = create_disk_with_membrane(radius=radius, h=h).submesh
    return Geometry(name=name, subdomains={surface_subdomain: SubdomainGeometry(mesh=submesh, kind="surface")})


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
    """The §1.6.6 coupled geometry: an annular `extracellular` bulk whose inner
    boundary is the `membrane` surface subdomain (coupled at `interface`) and whose
    outer boundary is `outer` (the reservoir)."""

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
    compartment and outer-annulus `outer_subdomain`, meeting at the `membrane`. `interface` labels the
    shared membrane curve, `outer` the external (reservoir) circle. Carries the three entity maps so a
    coupling form can reference both bulk traces at the membrane."""

    cell = create_cell_extracellular(inner_radius=inner_radius, outer_radius=outer_radius, h=h)
    return InterfaceCoupledGeometry(
        name=name,
        inner_subdomain=inner,
        outer_subdomain=outer_subdomain,
        membrane_subdomain=membrane,
        inner_mesh=cell.cytosol_mesh,
        outer_mesh=cell.extracellular_mesh,
        membrane_mesh=cell.membrane_mesh,
        inner_entity_map=cell.cytosol_entity_map,
        outer_entity_map=cell.extracellular_entity_map,
        membrane_entity_map=cell.membrane_entity_map,
        parent_mesh=cell.parent_mesh,
        cell_tags=cell.cell_tags,
        facet_tags=cell.facet_tags,
        inner_region_tag=cell.inner_region_tag,
        outer_region_tag=cell.outer_region_tag,
        interface=interface,
        interface_tag=MEMBRANE_TAG,
        outer=outer,
        outer_tag=OUTER_TAG,
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
