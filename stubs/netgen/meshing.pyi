"""Minimal local stub for the netgen.meshing subset vcell-fenics uses (Netgen ships
pybind11 runtime signatures but no static types). Scoped to our call sites. Discovered
API notes so the next reader doesn't re-derive them:

- ``FaceDescriptor(surfnr, domin, domout, bc)`` — ``domin``/``domout`` are the domain
  numbers in front of / behind the surface (0 = void); a merged surface mesh is
  partitioned into subdomains this way for ``GenerateVolumeMesh`` (the 3D analog of a 2D
  ``SplineGeometry`` leftdomain/rightdomain). See ``backend/realize._mesh_box_with_surfaces``.
- To volume-mesh from surfaces, the *outer* box surface must come from Netgen's own mesher
  (CSG ``OrthoBrick``): it carries the local mesh-size function that raw triangles lack.
  Raw marched triangles as an interface produce slivers — re-mesh them via ``STLGeometry``.
- ``Mesh.CalcLocalH(grading, layer)`` takes TWO args (``layer`` defaults to 1; passing 0
  segfaults) — only needed for hand-built surfaces, which we avoid.
"""

from collections.abc import Sequence
from typing import overload

class MeshVertex:
    nr: int

class MeshPoint:
    p: tuple[float, float, float]

class FaceDescriptor:
    def __init__(self, *, surfnr: int = ..., domin: int = ..., domout: int = ..., bc: int = ...) -> None: ...

class Element2D:
    vertices: Sequence[MeshVertex]
    index: int
    def __init__(self, fd: int, vertices: Sequence[MeshVertex]) -> None: ...

class Element3D:
    vertices: Sequence[MeshVertex]
    index: int

class Mesh:
    def __init__(self, dim: int = ...) -> None: ...
    def SetMaterial(self, domain: int, name: str) -> None: ...
    @overload
    def Add(self, obj: FaceDescriptor) -> int: ...
    @overload
    def Add(self, obj: MeshPoint) -> MeshVertex: ...
    @overload
    def Add(self, obj: Element2D) -> None: ...
    def __getitem__(self, vertex: MeshVertex) -> MeshPoint: ...
    def GenerateVolumeMesh(self) -> None: ...
    def Points(self) -> Sequence[MeshPoint]: ...
    def Elements2D(self) -> Sequence[Element2D]: ...
    def Elements3D(self) -> Sequence[Element3D]: ...
