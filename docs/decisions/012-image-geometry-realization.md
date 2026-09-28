# ADR 012 — Realizing image geometries: smoothed label field, conforming multi-label boundaries, Netgen

**Date:** 2026-09-23
**Status:** Accepted
**Amends:** [ADR 007](007-geometry-source-of-truth.md) (the image now carries its voxels), [ADR 008](008-gmsh-license-isolation.md) §8 (a
direct labelled-surface route replaces the per-surface STL recipe for images)

## Context

About 16% of VCell geometries in the corpus (926 of them) are **image-based**: a segmented uint8 image
whose pixel values are subvolumes (ec / cytosol / nucleus …). Until now vcell-fenics imported their
metadata and refused to mesh them. Real segmented images are harder than analytic shapes in three ways:
- regions are cut by the **image edge**;
- three or more regions meet at **junctions**, e.g. two touching cells in extracellular space;
- the boundary is a **pixel staircase**, often with anisotropic voxels (the tutorial image is
  0.29 × 0.29 × 0.79 µm).

The analytic path marches each subvolume's implicit field separately and embeds the surfaces through a
per-surface STL remesh. That handles none of the three: surfaces can't share vertices along a junction
curve, and it refuses anything touching the box.

VCell's own finite-volume path samples the image onto the simulation grid (nearest pixel) and uses voxel
faces as membranes, with areas taken from a Taubin-smoothed surface. Here the aim is **body-fitted,
smoothed meshes**, realizing the geometry better than the staircase (ADR 007), with **any topology**.

## Decision

Image realization is a three-stage pipeline behind the existing seam (`_realize_image_partition` in
`backend/realize.py` returns the same `(parent mesh, _Tagging)` as the analytic partitions). So
`realize`, `realize_interface_coupled`, the solvers and the results bundle are unchanged.

1. **Voxels in the formalism.** `GeometryImage.compressed_content` carries VCell's own encoding: the hex
   of a zlib stream of `nx·ny·nz` uint8 values, x-fastest. It is lossless and a YAML stays
   self-contained. VCell's lattice is **vertex-centred**: pixel `i` sits at
   `origin + i·extent/(n − 1)`, so the first and last pixels lie on the box faces.
2. **The label field** (`backend/labels.py`).
   - Analytic subvolumes are rasterized over the image, the earliest winning (VCell's priority).
   - Each subvolume's indicator is Gaussian-smoothed, with **σ = 1 pixel per axis**, so anisotropic
     voxels are smoothed by their own spacing.
   - The smoothed indicators are resampled trilinearly onto an **≈ h lattice** whose nodes include the
     box faces, and each node takes the subvolume with the largest value.
   - Clean-up then absorbs specks that smoothing severed (pieces beyond the image's own count, and under
     2% of the subvolume), and removes **pinches**. A pinch is a subvolume touching itself only at an
     edge or a corner, which would make the surface non-manifold; each is removed by the best local
     relabelling.
   - A subvolume that **vanishes** at this h is an error. Pieces that merge or split, and touching pairs
     with no SurfaceClass, are `ImageGeometryWarning`s that the runner logs.
3. **Conforming boundaries** (`backend/label_surfaces.py`).
   - **VTK SurfaceNets** (2D and 3D) extracts every touching label pair at once, so boundaries meet
     conformingly at junction curves and points.
   - The grid is padded with **one sentinel label per box face**, which turns box cuts, edges and corners
     into ordinary junctions. Those vertices are snapped exactly onto their planes.
   - Each element carries its two sides, with the normal pointing into the second. In 3D that is VTK's
     convention; 2D segments are oriented by a grid lookup.
   - Smoothing is deterministic and constrained: Taubin passes move a vertex by its class (interface, box
     face, junction curve or box edge, fixed).
   - Vertices are then **projected** by Gauss–Newton onto the smooth interfaces I_a = I_b (and I_b = I_c
     at junctions), guarded against far targets and flipped elements.
4. **Netgen** (`realize.py`), meshing serially on rank 0 as before.
   - **2D:** every labelled segment is a `SplineGeometry` segment between shared points.
   - **3D:** the labelled triangles are the surface mesh, loaded in bulk with one `FaceDescriptor` per
     region pair, then `GenerateVolumeMesh`.
   - **Netgen's domain is the region**, so no classification is needed, only the (now vectorized) facet
     tagging.
   - If Netgen rejects the smoothed boundary as overlapping (a thin region's surfaces within a cell of
     each other), the realizer **falls back** to the unprojected boundary, then to SurfaceNets' own. That
     last one has one vertex per lattice cell and can't intersect itself. Each fallback is a warning.
   - A 3D size past about 4M tetrahedra is refused, with the coarsest usable h.

## Consequences

- **Accuracy is the image's own.** The residual error is the image's rasterization plus the smoothing's
  curvature shrink (≈ σ²/R in radius), both O(pixel). Cutting by the box adds none.
- **On VCell's tutorial image at h = 1 µm:**
  - region volumes are within 0.3 / 1.8 / 2.3% of VCell's stored values;
  - membrane areas are within 3.9 / 5.2% of VCell's voxel-surface areas.
- **Against fvsolver:** nucleocytoplasmic exchange on that image agrees to 1.7% in the nuclear filling
  curve and 0.8% relative L2 in the cytosolic field, and both conserve to 1e-14
  (`cross_validation/README.md`).
- **Membranes are open patches.** A membrane between two subvolumes that also meet a third ends on a
  junction curve, which surface PDEs see as a natural zero-flux edge.
- **A latent bug fixed:** `realize_interface_coupled` tagged every exterior facet as the outer
  reservoir wall. It is now only the outer compartment's share, which matters once an inner region
  touches the box.
- **Rejected alternatives:**
  - per-subvolume marching plus STL remesh (no conforming junctions);
  - Netgen `STLGeometry` on the whole multi-label surface (it meshes one solid, with no domin/domout);
  - a voxel-staircase mesh (it inflates membrane areas by about 27% in 2D and 50% in 3D, and biases
    membrane fluxes).
- **Later: analytic subvolumes that touch the box (#187)** reuse this pipeline. Their label grid is the
  priority rasterization on an ``h`` lattice (`analytic_label_geometry`), with the priority-resolved
  implicit functions as exact indicators, so the projection lands on the analytic surfaces. Three guards
  came with it: a projected vertex stays a quarter cell off the box faces (a predicate such as
  `z >= z0`, with `z0` the box's own face, has a zero set on the face that no interface can lie on); a
  boundary with collapsed elements is refused before Netgen, which otherwise aborts the process; and a
  Netgen result missing a region is an error rather than a silently empty compartment. Each refusal
  falls back to the next surface level.
- **Not yet:** images in moving-boundary applications (the ALE remesh path is analytic-free but
  untested on images), and mixed analytic + image geometries in VCell's own models. The rasterization
  supports them, but no fixture covers them yet.
