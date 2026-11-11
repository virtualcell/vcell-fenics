# Spike: libsupermesh as the conservative transfer kernel (2026-11-11)

Question (Jim): ADR 016's transfer is vcell-fenics' own pure-Python supermesh (`core/bulk_remap.py`), about
75 s per remesh at 133k cells, and 2D only. Would libsupermesh (Firedrake's library, LGPL-2.1) do it faster,
and in 3D?

## Build (macOS arm64, measured)

- Not in DOLFINx 0.10 and not on conda-forge; on PyPI as an sdist only (`libsupermesh` 2026.3, scikit-build-core,
  depends on `rtree`, which bundles libspatialindex).
- Needs MPI with Fortran (`find_package(MPI REQUIRED)`) and a Fortran compiler. vcell-fenics' env has MPICH's
  `mpif90` but not the conda gfortran it wraps; the spike used Homebrew gfortran 15.2 via `MPICH_FC`, and Apple
  clang via `MPICH_CC` / `MPICH_CXX` (the env has no `clang++`).
- `MPICH_FC=gfortran MPICH_CC=clang MPICH_CXX=clang++ pip install libsupermesh -C cmake.define.CMAKE_Fortran_COMPILER=$ENV/bin/mpif90 ...`
  builds in about 10 s. A manual CMake build is the same plus copying libspatialindex next to the library.

## Kernel

`mixed_mass.c`: the P1 mixed mass matrix `B[j_new, i_old] = ∫ φ_new_j φ_old_i` over the supermesh, in 2D and
3D, from libsupermesh's R-tree finder (`libsupermesh_tree_intersection_finder_*`) and
`libsupermesh_intersect_tris_real` / `_tets_real`; exact on each piece, since for P1 functions on a simplex S,
∫ f g = |S| / ((d+1)(d+2)) (Σ f_k g_k + Σ f_k Σ g_k). Projection: `M_new c_new = B c_old` (CG). Build:
`clang -O2 -shared -fPIC -I$P/include -o libmixedmass.dylib mixed_mass.c -L$P/lib -lsupermesh`.

## Results (`bench.py`; unit square / cube, two unrelated meshes, interior nodes perturbed; serial)

| Case | Old / new cells | Finder | Assemble | Solve | Total change | Linear field error |
|---|---|---|---|---|---|---|
| 2D small, vs `supermesh_project_2d` | 3,200 / 2,738 | – | 0.012 s all | – | 0 | 5e-13; agrees with the Python kernel to 1.3e-13 (Python: 1.54 s) |
| 2D | 133,128 / 126,002 | 0.08 s | 0.37 s | 0.06 s | 2e-16 | 3e-12 |
| 3D | 48,000 / 34,992 | 0.10 s | 2.2 s | 0.02 s | 0 | 5e-12 |
| 3D | 131,712 / 105,456 | 0.27 s | 6.4 s | 0.07 s | 3e-16 | 7e-12 |

The covered measure matches the domain to 1e-13, so no piece is lost: no clipping cutoff is needed for
conservation (the kernel's `rel_tol` was 1e-14).

Limits of the spike: synthetic meshes of a box, not a partition's compartments (the real transfer runs per
compartment, on submeshes); the kernel stores (d+1)² entries per candidate pair, including pairs that do
not intersect, so a 384k-tet case ran out of memory: skipping empty pairs fixes that. Serial only.
