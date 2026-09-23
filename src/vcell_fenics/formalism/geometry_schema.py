"""The geometry formalism — the `GeometryDescription` dataclass tree.

The declarative source of truth for a spatial domain (ADR 007;
`docs/modeling/geometric-formalism.md`), peer to the `MathDescription`. A bounding domain
partitioned into named **subvolumes** (volume regions) and **surface classes** (membranes
between a pair of subvolumes), mirroring VCell's `pyvcell.vcml.models_geometry.Geometry`.

The names are the contract the MathDescription binds to (`cross_validate`): a subvolume name
is a `volume` subdomain, a surface name is a `surface` subdomain. The concrete mesh is a
*derived realization* (`backend/geometry.py`), not part of this spec.

An `image`-typed subvolume references a pixel class by value; the `GeometryImage` carries the
image's size, classes and — when the source has it — its voxels, in VCell's own encoding (hex of a
zlib stream of uint8, x-fastest), so the image imports losslessly and a YAML stays self-contained.
Analytic expressions are in
the same §1.8 expression language as the math formalism (`geom.x[…]`), so the two share one
parser.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

# Attached to each dataclass below so a `pydantic.TypeAdapter` (the geometry_io loader)
# rejects unknown fields at the YAML/JSON boundary. The classes stay plain stdlib
# dataclasses — this is only consulted when pydantic validates them; nothing else changes.
_FORBID_EXTRA = ConfigDict(extra="forbid")

# A subvolume's membership rule (VCell's `SubVolumeType`):
#   compartmental — the whole non-spatial (dim-0) domain; no payload
#   analytic      — the region where `expression`, a boolean predicate over geom.x, is true
#                   (lowered to an inside-negative implicit function by `rvachev.py`)
#   csg           — a constructive-solid-geometry shape (payload not modelled yet; see ADR 007)
#   image         — the voxels of the geometry's image carrying `pixel_value`
SubVolumeType: TypeAlias = Literal["compartmental", "analytic", "csg", "image"]


@dataclass(frozen=True, slots=True)
class PixelClass:
    """A labelled voxel value in a segmented image."""

    __pydantic_config__ = _FORBID_EXTRA
    name: str
    pixel_value: int


@dataclass(frozen=True, slots=True)
class GeometryImage:
    """A segmented image backing `image`-typed subvolumes: its `size` ``(nx, ny, nz)``, its pixel
    classes, and its voxels as ``compressed_content`` — VCell's own encoding, the hex of a zlib stream
    of ``nx·ny·nz`` uint8 pixels indexed ``x + nx·(y + ny·z)`` (``None`` when the source carried only
    metadata). VCell's lattice is **vertex-centred**: pixel ``i`` sits at
    ``origin + i·extent / (n − 1)``, so the first and last pixels lie on the domain boundary."""

    __pydantic_config__ = _FORBID_EXTRA
    name: str
    size: tuple[int, int, int]
    pixel_classes: tuple[PixelClass, ...] = ()
    compressed_content: str | None = field(default=None, repr=False)

    def voxels(self) -> NDArray[np.uint8]:
        """The pixels as a ``(nz, ny, nx)`` uint8 array (decoded on each call). Raises ``ValueError``
        if the image carries no voxels or they do not decode to ``nx·ny·nz`` bytes."""

        if not self.compressed_content:
            raise ValueError(f"image {self.name!r} carries no voxel data")
        try:
            raw = zlib.decompress(bytes.fromhex(self.compressed_content))
        except (ValueError, zlib.error) as exc:
            raise ValueError(f"image {self.name!r}: voxel data is not hex-encoded zlib ({exc})") from exc
        nx, ny, nz = self.size
        if len(raw) != nx * ny * nz:
            raise ValueError(f"image {self.name!r}: {len(raw)} voxels decoded, size {self.size} needs {nx * ny * nz}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(nz, ny, nx)

    @classmethod
    def from_voxels(
        cls, name: str, voxels: NDArray[np.uint8], pixel_classes: tuple[PixelClass, ...] = ()
    ) -> GeometryImage:
        """Encode a ``(nz, ny, nx)`` (or 2D ``(ny, nx)``) uint8 label array the way VCell stores it."""

        array = np.asarray(voxels, dtype=np.uint8)
        if array.ndim == 2:
            array = array[np.newaxis]
        if array.ndim != 3:
            raise ValueError(f"image voxels must be 2D or 3D, got shape {array.shape}")
        nz, ny, nx = array.shape
        content = zlib.compress(np.ascontiguousarray(array).tobytes()).hex().upper()
        return cls(name=name, size=(nx, ny, nz), pixel_classes=pixel_classes, compressed_content=content)


@dataclass(frozen=True, slots=True)
class SubVolume:
    """A named volume region (a `volume` subdomain), defined by its `type` (§1.2). `expression`
    is the boolean predicate for `analytic` (the region where it is true; lowered to an implicit
    function by `rvachev.py`); `pixel_value` is the image class for `image`; `compartmental` and
    `csg` carry neither."""

    __pydantic_config__ = _FORBID_EXTRA
    name: str
    type: SubVolumeType
    expression: str | None = None
    pixel_value: int | None = None


@dataclass(frozen=True, slots=True)
class SurfaceClass:
    """A membrane (a `surface` subdomain): the interface between an ordered pair of subvolumes.
    The order `(inside, outside)` fixes the outward normal and the inside/outside trace
    directions for cross-membrane coupling."""

    __pydantic_config__ = _FORBID_EXTRA
    name: str
    inside: str
    outside: str


@dataclass(frozen=True, slots=True)
class GeometryDescription:
    """A named spatial domain: a bounding box (`dim`, `extent`, `origin`) partitioned into
    `subvolumes` with `surfaces` between them. `dim = 0` is the non-spatial / well-mixed case
    (one `compartmental` subvolume, no surfaces)."""

    __pydantic_config__ = _FORBID_EXTRA
    name: str
    dim: int
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    subvolumes: tuple[SubVolume, ...] = ()
    surfaces: tuple[SurfaceClass, ...] = ()
    image: GeometryImage | None = field(default=None)
