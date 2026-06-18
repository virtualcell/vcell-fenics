"""The geometry formalism — the `GeometryDescription` dataclass tree.

The declarative source of truth for a spatial domain (ADR 007;
`docs/modeling/geometric-formalism.md`), peer to the `MathDescription`. A bounding domain
partitioned into named **subvolumes** (volume regions) and **surface classes** (membranes
between a pair of subvolumes), mirroring VCell's `pyvcell.vcml.models_geometry.Geometry`.

The names are the contract the MathDescription binds to (`cross_validate`): a subvolume name
is a `volume` subdomain, a surface name is a `surface` subdomain. The concrete mesh is a
*derived realization* (`backend/geometry.py`), not part of this spec.

This is metadata only: an `image`-typed subvolume references a pixel class by value, and the
`GeometryImage` carries the image's size and classes — but **not** the raw voxel blob (it is
large and stays in the source; image meshing is a later increment). Analytic expressions are in
the same §1.8 expression language as the math formalism (`geom.x[…]`), so the two share one
parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

# A subvolume's membership rule (VCell's `SubVolumeType`):
#   compartmental — the whole non-spatial (dim-0) domain; no payload
#   analytic      — the region where `expression` (an implicit function of geom.x) is inside
#   csg           — a constructive-solid-geometry shape (payload not modelled yet; see ADR 007)
#   image         — the voxels of the geometry's image carrying `pixel_value`
SubVolumeType: TypeAlias = Literal["compartmental", "analytic", "csg", "image"]


@dataclass(frozen=True, slots=True)
class PixelClass:
    """A labelled voxel value in a segmented image."""

    name: str
    pixel_value: int


@dataclass(frozen=True, slots=True)
class GeometryImage:
    """Metadata for a segmented image backing `image`-typed subvolumes. The raw voxel data is
    deliberately not carried here (see the module docstring)."""

    name: str
    size: tuple[int, int, int]
    pixel_classes: tuple[PixelClass, ...] = ()


@dataclass(frozen=True, slots=True)
class SubVolume:
    """A named volume region (a `volume` subdomain), defined by its `type` (§1.2). `expression`
    is the implicit analytic function for `analytic`; `pixel_value` is the image class for
    `image`; `compartmental` and `csg` carry neither."""

    name: str
    type: SubVolumeType
    expression: str | None = None
    pixel_value: int | None = None


@dataclass(frozen=True, slots=True)
class SurfaceClass:
    """A membrane (a `surface` subdomain): the interface between an ordered pair of subvolumes.
    The order `(inside, outside)` fixes the outward normal and the inside/outside trace
    directions for cross-membrane coupling."""

    name: str
    inside: str
    outside: str


@dataclass(frozen=True, slots=True)
class GeometryDescription:
    """A named spatial domain: a bounding box (`dim`, `extent`, `origin`) partitioned into
    `subvolumes` with `surfaces` between them. `dim = 0` is the non-spatial / well-mixed case
    (one `compartmental` subvolume, no surfaces)."""

    name: str
    dim: int
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    subvolumes: tuple[SubVolume, ...] = ()
    surfaces: tuple[SurfaceClass, ...] = ()
    image: GeometryImage | None = field(default=None)
