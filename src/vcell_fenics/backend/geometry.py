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

from dataclasses import dataclass

from dolfinx.mesh import Mesh

from vcell_fenics.approaches.static.geometry import create_disk
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
class Geometry:
    """A named geometry: subdomain-class name → its mesh and kind."""

    name: str
    subdomains: dict[str, SubdomainGeometry]

    def kind_of(self, subdomain: str) -> SubdomainKind | None:
        entry = self.subdomains.get(subdomain)
        return entry.kind if entry is not None else None

    def mesh_of(self, subdomain: str) -> Mesh:
        return self.subdomains[subdomain].mesh


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


def make_disk_geometry(name: str, *, volume_subdomain: str, radius: float = 1.0, h: float = 0.1) -> Geometry:
    """A bundled 2D disk exposed as a single `volume` subdomain class."""

    mesh = create_disk(radius=radius, h=h).mesh
    return Geometry(name=name, subdomains={volume_subdomain: SubdomainGeometry(mesh=mesh, kind="volume")})


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
    return diagnostics
