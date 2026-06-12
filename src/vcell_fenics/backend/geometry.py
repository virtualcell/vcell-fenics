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
from dolfinx.mesh import Mesh, MeshTags
from numpy.typing import NDArray

from vcell_fenics.approaches.multicompartment.geometry import (
    MEMBRANE_TAG,
    OUTER_TAG,
    create_cell_extracellular,
)
from vcell_fenics.approaches.static.geometry import BOUNDARY_TAG, create_disk
from vcell_fenics.approaches.submesh.geometry import create_disk_with_membrane
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
