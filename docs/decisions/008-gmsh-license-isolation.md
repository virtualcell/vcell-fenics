# ADR 008 — The mesher is Netgen (LGPL) at the import boundary; gmsh isolation is a contingency, not the plan

**Date:** 2026-07-13
**Status:** Proposed
**Supersedes:** the initial framing of this ADR (isolate GPL gmsh behind a process/service
boundary). That design is retained below as the **contingency fallback** (§7), not the primary plan.

## Context

`vcell-fenics` is meant to be hostable as a backend PDE service for VCell, and its higher layers
(formalism, `pyvcell_bridge`, IO, viz, analysis) to run inside or beside pyvcell. VCell and pyvcell
are **MIT**, so the license reach of the meshing dependency matters. Licenses read from the installed
conda packages:

| Component | License | Copyleft |
|---|---|---|
| fenics-dolfinx / ufl / ffcx | LGPL-3.0-or-later | weak — import boundary is enough |
| basix, dolfinx_mpc, scifem, pyvista | MIT | none |
| petsc / petsc4py | BSD-2-Clause | none |
| **gmsh / python-gmsh** | **GPL-2.0-or-later** | **strong** |
| **netgen / pyngcore** | **LGPL-2.1-only** | **weak — import boundary is enough** |

The original decision recorded here solved the gmsh problem *by isolation*: gmsh is GPL, so route all
meshing through a neutral `Mesher` port and reach gmsh only across a process/service boundary
(subprocess or remote), with a gmsh-free XDMF interchange, plus a permissive 2D `VtkMesher` and a note
that 3D body-fitted meshing had no permissive option. That design works but is heavy: an IPC boundary
on the in-loop remesh, a mesher we hand-roll in VTK, and an unresolved 3D gap.

**What changed:** Netgen (the NGSolve mesher) is **LGPL-2.1** — the *same weak-copyleft tier as
DOLFINx itself*. An `import netgen` from MIT/permissive code carries **no** copyleft on the caller, so
Netgen can be called **in-process at the import boundary with no isolation at all**. It is on
conda-forge for our two locked platforms, ships a Python API, does quality 2D *and* 3D meshing (with
Steiner points) plus OCC CSG, and its point/cell arrays feed `dolfinx.mesh.create_mesh` directly (no
`.msh`, no ngsPETSc). If Netgen covers the meshing needs, the entire isolation apparatus becomes
unnecessary — which a spike (commit `cf74409`) confirms it does.

## Decision

**Netgen (LGPL-2.1) is the mesher, called in-process at the import boundary. No process/service
isolation, no plugin, and no gmsh-free interchange format is required for licensing — LGPL does not
propagate copyleft to callers, exactly as DOLFINx does not.** The mesh is built from Netgen's
point/cell arrays via `create_mesh` directly.

### 1. Validated by spike (`cf74409`)

- **Env coexistence.** `netgen 6.2.2602` (native py3.14 conda-forge build) co-solves and imports
  alongside the pinned `fenics-dolfinx 0.10` / MPICH env.
- **2D remesh** (`core.region_remesh_netgen.mesh_region_netgen`, the LGPL analogue of the gmsh
  `mesh_region`): on a non-convex starfish loop — area preserved to **2.7e-16**, boundary conforms to
  the polyline to **1.1e-16** (`Γ_new ⊂ Γ_old`), **min angle ~34°, zero slivers**. A test mirroring
  `test_core_region_remesh.py` passes 7/7.
- **3D quality tets** on the same dumbbell (two overlapping spheres, concave neck) the `VtkMesher`
  path failed: **min dihedral 17.6°, zero slivers, zero convex-hull leaks**, DOLFINx 3D mesh built
  directly. This is the capability the isolation design had *no* permissive answer for (old §8a).

### 2. gmsh is demoted; `VtkMesher` is dropped

- **`VtkMesher` (old §8) is removed from the plan.** It existed only because gmsh was GPL and VTK was
  permissive; Netgen is permissive *and* a real quality mesher covering 2D **and** 3D, so the
  hand-rolled `vtkDelaunay2D` + seeding path buys nothing. (VTK's `vtkMeshQuality` remains a handy
  permissive quality-gating tool, unrelated to meshing.)
- **gmsh is a contingency, not a dependency of the design.** It stays only for anything Netgen cannot
  yet do; if that set is ever non-empty and non-migratable, the §7 isolation design is how gmsh ships.

### 3. Operational constraints (learned in the spike — these are load-bearing)

- **Cap Netgen's threads.** Its default multi-threaded TaskManager keeps a worker pool that
  busy-waits in a long-lived process (pinned cores, stalled progress under a pytest session). Call
  `pyngcore.SetNumThreads(1)` before the mesher loads; our serial 2D/3D remeshes gain nothing from
  threads. (Done in `region_remesh_netgen`.)
- **Do not interleave gmsh-OCC meshing and Netgen in one process.** The spike saw instability when a
  long-lived process mixed the two (could not be reduced to a clean minimal repro — treat as real
  until disproven). Consequences: `region_remesh_netgen` is **opt-in** (not re-exported from
  `vcell_fenics.core`, so gmsh-only paths never load Netgen), its test is **gmsh-free**, and — see §4
  — `realize()` should migrate off gmsh onto Netgen so a solver process uses **one** mesher.

### 4. Migrate `realize()` off gmsh onto Netgen — **2D done**

**2D `realize()` is migrated** (PR #108): `_mesh_box_with_contours` builds a Netgen `SplineGeometry`
with each marched contour as a conforming internal boundary and leftdomain/rightdomain from the
contours' containment forest, so Netgen's per-element material index is the region tag
`_classify_and_tag` consumes. gmsh is gone from `realize.py`; the full non-integration suite (615
tests) passes with gmsh and Netgen coexisting in one process. The `approaches/*/geometry.py`
prototypes still use gmsh (separate, low-priority migrations). Until fully migrated, keep gmsh-based
paths and Netgen in **separate processes** (§3).

*Follow-up (2026-10-08):* the migration is complete. `realize()` is Netgen in 2D and 3D (the 3D
multi-region path is productized, and image geometries follow ADR 012); the ALE driver's remesh
call went to `mesh_region_netgen` and, in 3D, `backend/remesh_3d.py`; the `approaches/*` prototypes
were removed from `src/` (their gmsh meshers live under `tests/gmsh_meshers/`); `src/` imports no
gmsh, so the separate-process rule applies only to the dev/test suite.

### 5. `fix_boundary_nodes` fast path is not available from high-level Netgen

Netgen's high-level 2D mesher always resamples the boundary at `h` (verified: 48 boundary nodes from a
12-vertex input, regardless of per-segment `maxh` / `MeshingParameters`). The exact-vertex
preservation that `fix_boundary_nodes=True` needs — the interior-only fast path that lets
`correct_surface_trace` be skipped — has **no high-level equivalent**; it would require Netgen's
low-level `Element1D` construction. `mesh_region_netgen` raises `NotImplementedError` for it rather
than silently resampling. Deferred; the default resample-and-full-remap path is correct meanwhile.

### 6. Near-pinch robustness — still to check

Netgen is a mature mesher, but its behavior on the ALE driver's hardest inputs (very thin necks /
near-pinch-off boundaries) has not been stressed here. Gate any switch of the ALE driver's remesh call
from gmsh to Netgen on that check.

*Follow-up (2026-10-08):* the switch happened and the check was done the hard way on the 3D
cleavage-furrow runs (integration tracker, "3D moving boundaries"): a neck thinner than the mesh can
resolve raises `PinchOffError` (2D: `local_thickness` in `region_remesh_netgen`; 3D: `remesh_3d`), a
partial Netgen fill is rejected by an enclosed-volume check, and the 3D remesher falls back through
surfaces of decreasing fidelity before giving up with a dated, located message.

### 7. Contingency: the gmsh-isolation design (retained, not primary)

If Netgen is ever disqualified (a robustness or packaging failure §6 might surface), gmsh ships behind
the original design, summarized so it survives:

- A neutral `Mesher` port; gmsh reached only across a **process boundary** (local subprocess or remote
  service) via a **gmsh-free XDMF interchange** (both `dolfinx.io.gmsh` readers reload gmsh
  in-process, so the `gmsh→dolfinx` conversion happens inside the gmsh-side process, which emits XDMF
  the solver reads with `dolfinx.io.XDMFFile`). One `MesherClient` adapter, two transports;
  `vcell-fenics` stays independently useful with no mesher configured; the two packages ship
  separately so no combined GPL artifact is distributed. This is only needed because gmsh is GPL — it
  is **not** needed for Netgen.

### 8. 3D derisk — Netgen is the mesher for 2D and 3D (single- and multi-region)

A 3D spike (not committed) verified the netgen-3D unknowns that made 3D look risky, including the
multi-region case that first appeared to be a blocker:

- **Netgen 3D mesher is graceful.** Analytic CSG (sphere-in-box; nested nucleus/cytosol/ecm spheres)
  meshes cleanly with the tets+region-tags → DOLFINx bridge (`el.index` realigned via
  `topology.original_cell_index`), min dihedral 12–18°, zero slivers, no hangs.
- **Marched-surface → volume (single region) works, and beats gmsh here.** Marching-cubes →
  triangulated STL → Netgen `STLGeometry.GenerateMesh` on a sphere and a **thin-neck dumbbell**: min
  dihedral 20.8° / 14.6°, **zero slivers**. gmsh on the *same* STLs (out-of-box, un-optimized)
  produced slivers on the dumbbell (min dihedral 4.8°, 21 slivers, 3× the tets). So the "netgen may
  not be graceful in 3D" worry is retired — on this pipeline the gap runs the other way. (Fair caveat:
  gmsh's quality is likely improvable with its optimizer / tuned `classifySurfaces`.)
- **Multi-region marched-surface works in Netgen — resolved.** `realize()`'s 3D needs the box
  *partitioned* by the marched surface (interior + background, conforming interface; even a single cell
  is two regions). The **recipe** (3D analog of the 2D `SplineGeometry` leftdomain/rightdomain path):
  1. build the **box surface with Netgen CSG** (`OrthoBrick`) — this carries the correct orientation
     *and* the local mesh-size function that manual triangles lack;
  2. **re-mesh each marched implicit surface** through `STLGeometry.GenerateMesh` (marching-cubes → STL
     → quality surface triangulation) — using the *raw* marched triangles as the interface produces
     slivers, re-meshing gives clean tets;
  3. merge the surfaces into one `Mesh` with a `FaceDescriptor` per surface (`domin`/`domout` from the
     contours' containment forest — box `domin=1,domout=0`; a nested surface `domin=child,domout=parent`);
  4. `GenerateVolumeMesh()`; `el.index` is the per-region tag, realigned to DOLFINx via
     `topology.original_cell_index` (as in 2D).

  Verified on box-plus-sphere: correct conforming region volumes (inside 4.11 vs 4.19, bg 28.66 vs
  28.58), **min dihedral ~15.5°, zero slivers**, controllable tet count — *better* quality than gmsh's
  two-volume approach (min dihedral 10.6°) on the same surface. The earlier "fragile" reading was two
  fixable mistakes (hand-triangulated box → wrong orientation/no `localh`; raw marched interface →
  slivers), not a Netgen limitation.

  **Decision:** Netgen is the mesher for **2D, 3D single-region, and 3D multi-region** — in-process,
  LGPL, no gmsh needed for 3D. gmsh stays only as the §7 contingency. (Follow-up: the multi-region 3D
  path above is spiked, not yet productionized in `realize()`; that is the next 3D implementation task.
  *Done since* — `realize()` meshes 3D multi-region analytic and image geometries with Netgen, ADR 012.)

## Consequences

**Positive:**

- **No GPL anywhere on the meshing path**, and no isolation machinery — the solver links only
  weak-copyleft (DOLFINx, Netgen) and permissive libraries, so VCell stays MIT and the higher layers
  run freely beside pyvcell. Legally this is the *same settled posture as depending on DOLFINx*, not
  the disputed GPL-plugin grey zone the isolation design had to navigate.
- **2D and 3D covered by one permissive, quality mesher**, in-process, with a direct DOLFINx bridge —
  simpler and more capable than gmsh-isolation + `VtkMesher` + an open 3D gap.

**Negative / costs:**

- **We own a mesher *integration*** (not the mesher itself): the thread cap, the opt-in import, and
  the gmsh/Netgen non-coexistence rule are all real constraints to respect.
- **`realize()` must migrate off gmsh** (§4) to actually drop gmsh and to avoid the coexistence
  hazard; until then two meshers exist in the codebase and must be kept in separate processes.
  *(Done — see the §4 follow-up; `src/` is gmsh-free.)*
- **`fix_boundary_nodes` gap** (§5) is open and costs nothing today (the driver never takes that
  path); **near-pinch robustness** (§6) was settled by the pinch-off guards and fallbacks.

**Neutral:**

- gmsh's GPL is no longer a *design driver*; it is a contingency (§7). ADR 007 stands — the geometry
  formalism is the source of truth; this ADR only changes which engine realizes/remeshes it.

## History

This ADR originally decided to isolate GPL gmsh behind a `Mesher` port and a process/service boundary,
with a permissive `VtkMesher` for 2D and gmsh-only for 3D. The Netgen spike (`cf74409`) showed an
LGPL mesher covers 2D **and** 3D in-process with no isolation, so the decision was rewritten around
Netgen and the isolation design demoted to the §7 contingency.

## Amendment (2026-09-23) — image geometries mesh from their own labelled surfaces

For image geometries the §8 recipe (a CSG box, each marched surface re-meshed through `STLGeometry`, merged
under `FaceDescriptor`s) is replaced by a direct route: the conforming multi-label SurfaceNets triangles
*are* the surface mesh, loaded in bulk under one `FaceDescriptor` per region pair, and `GenerateVolumeMesh`
fills the domains ([ADR 012](012-image-geometry-realization.md)). Re-meshing each surface separately would
break conformity along junction curves, where three regions meet. Still Netgen, still serial on rank 0, still
gmsh-free.
