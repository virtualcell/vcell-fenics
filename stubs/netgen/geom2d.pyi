"""Minimal local stub for the netgen.geom2d subset vcell-fenics uses (2D meshing).
Scoped to `SplineGeometry` as called by `core.region_remesh_netgen` and
`backend.realize`. `Append`'s segment is `["line", p0, p1]` (a str tag + point ids
from `AppendPoint`); `leftdomain`/`rightdomain` assign the domain on each side of the
curve (the per-cell material index, read as `Element2D.index`)."""

from collections.abc import Sequence

from netgen.meshing import Mesh

class SplineGeometry:
    def __init__(self) -> None: ...
    def AppendPoint(self, x: float, y: float) -> int: ...
    def Append(
        self,
        segment: Sequence[object],
        *,
        leftdomain: int = ...,
        rightdomain: int = ...,
        bc: str | int = ...,
        maxh: float = ...,
    ) -> object: ...
    def GenerateMesh(self, *, maxh: float) -> Mesh: ...
