"""Minimal local stub for the netgen.stl subset vcell-fenics uses — re-meshing a marched
(marching-cubes) surface to a quality triangulation before it seeds a volume mesh
(backend/realize._mesh_box_with_surfaces)."""

from netgen.meshing import Mesh

class STLGeometry:
    def __init__(self, filename: str) -> None: ...
    def GenerateMesh(self, *, maxh: float) -> Mesh: ...
