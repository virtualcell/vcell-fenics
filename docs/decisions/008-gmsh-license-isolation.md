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

### 4. Migrate `realize()` off gmsh onto Netgen (follow-up)

Initial meshing (`backend/realize.py`, the `approaches/*/geometry.py` prototypes) still uses gmsh
today. Netgen has an OCC CSG kernel and does body-fitted 2D/3D, so this is migratable. Until it is
migrated, keep gmsh-based realization and Netgen remeshing in **separate processes** (per §3). Once
migrated, gmsh can be dropped from the runtime entirely.

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
- **`fix_boundary_nodes` gap** (§5) and **unverified near-pinch robustness** (§6) are open.

**Neutral:**

- gmsh's GPL is no longer a *design driver*; it is a contingency (§7). ADR 007 stands — the geometry
  formalism is the source of truth; this ADR only changes which engine realizes/remeshes it.

## History

This ADR originally decided to isolate GPL gmsh behind a `Mesher` port and a process/service boundary,
with a permissive `VtkMesher` for 2D and gmsh-only for 3D. The Netgen spike (`cf74409`) showed an
LGPL mesher covers 2D **and** 3D in-process with no isolation, so the decision was rewritten around
Netgen and the isolation design demoted to the §7 contingency.
