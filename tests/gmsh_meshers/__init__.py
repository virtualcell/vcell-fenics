"""gmsh-based reference meshers — test-only, kept out of `src/` for licensing (LICENSING.md).

gmsh is GPL-2.0-or-later, so it must not enter the distributed/runtime `vcell_fenics` package. These
meshers (formerly `vcell_fenics.approaches.*` and `vcell_fenics.core.region_remesh`) were relocated here
so the gmsh meshing know-how is preserved and the Netgen production path has a cross-check, while nothing
under `src/` imports gmsh. They run only in the `dev` test environment.

- `static.geometry.create_disk` — a body-fitted disk (gmsh OCC).
- `submesh.geometry.create_disk_with_membrane` — a disk plus its boundary submesh.
- `multicompartment.geometry` — `create_extracellular_annulus`, `create_cell_extracellular`, and the
  `make_cell_extracellular_geometry` / `make_extracellular_annulus_geometry` backend-`Geometry` builders
  (the annular-reservoir / concentric-annulus variants the box-bounded `realize_interface_coupled` does
  not reproduce exactly).
- `region_remesh.mesh_region` — the gmsh ALE region remesher (the LGPL analogue
  `vcell_fenics.core.region_remesh_netgen.mesh_region_netgen` is what the ALE driver uses at runtime).
"""
