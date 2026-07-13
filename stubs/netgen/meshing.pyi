"""Minimal local stub for the netgen.meshing subset vcell-fenics uses (Netgen ships
pybind11 runtime signatures but no static types). Scoped to our call sites — the
objects returned by ``SplineGeometry.GenerateMesh`` and the members we read off them.
Discovered API notes live here so the next reader doesn't re-derive them:

- ``FaceDescriptor(surfnr, domin, domout, bc)`` — ``domin``/``domout`` are the domain
  numbers in front of / behind the surface (0 = void); this is how a merged surface mesh
  is partitioned into subdomains for ``GenerateVolumeMesh`` (the 3D analog of a 2D
  ``SplineGeometry`` leftdomain/rightdomain). Not used by the committed 2D path yet;
  documented for the 3D multi-region realization (ADR 008 §8).
- ``Mesh.CalcLocalH(grading, layer)`` takes TWO args (``layer`` defaults to 1; passing 0
  segfaults). Not needed when the surface comes from Netgen's own mesher (it carries the
  local-size function already).
"""

from collections.abc import Sequence

class MeshVertex:
    nr: int

class MeshPoint:
    p: tuple[float, float, float]

class Element2D:
    vertices: Sequence[MeshVertex]
    index: int

class Mesh:
    def Points(self) -> Sequence[MeshPoint]: ...
    def Elements2D(self) -> Sequence[Element2D]: ...
