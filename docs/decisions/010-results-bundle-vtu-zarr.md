# ADR 010 — Results are a VTU + zarr bundle; moving meshes extend it with segments

**Date:** 2026-09-22
**Status:** Accepted (format choice confirmed by the user). The encoding details in §3 are pending
the step-2 spike; its results go in §6.
**Context doc:** [`docs/integration/vcell-solver-integration.md`](../integration/vcell-solver-integration.md)

## Context

Running vcell-fenics as a VCell solver ([ADR 011](011-vcell-solver-contract.md)) needs a durable
results format. Until now the CLI wrote a DOLFINx `XDMFFile` time series plus `summary.json`. The
readers the format must serve:

- **VCell's 3D field viewer.** `FieldViewerServer` (Java) decodes results into a JSON
  geometry/field contract for a vtk.wasm page. It already parses solver-produced `.vtu` meshes
  through `VtuGridParser`.
- **Remote viewing.** Results sit on the cluster's NFS and reach the client through VCell's data
  server; a future web viewer would fetch them over HTTP.
- **pyvcell.** It already stores finite-volume results as zarr (`(t, c, z, y, x)` plus
  `axes/channels/times` attrs).
- **ParaView and scientific users.**
- **VCell's live results view.** VCell shows partial results while a job runs, so the format has
  to be readable while it is being written.

What the format must represent:

- P1 point data on body-fitted simplicial meshes, in 2D and 3D.
- Several domains per run: compartment volumes and membrane surfaces (triangles in 3D, segments in
  2D).
- Time series.
- Moving boundaries: ALE runs (points move, topology fixed) and remeshing. The ALE remesh driver
  (`backend/ale.py`, `core/region_remesh_netgen.py`) already produces a sequence of segments with
  piecewise-constant topology.

Moving-boundary runs are the exception; the common case is a fixed mesh.

## Options considered

| Format | static | ALE | remesh | multi-domain | readers | assessment |
|---|---|---|---|---|---|---|
| **XDMF3 + HDF5** | ✓ | ✓ in the format; DOLFINx's `XDMFFile` writes the mesh once, so we'd need a custom writer | ✓ with a custom writer (new topology per segment) | one temporal collection per domain | ParaView, h5py/meshio, Java via `vcell-jhdf` + XML | FEniCS-community default and the previous CLI output. Two artifacts per domain; the XDMF library is dormant and Kitware's investment has moved to VTKHDF; HDF5 isn't chunk-addressable over HTTP and isn't safe to read while being written without SWMR discipline |
| **VTKHDF** | ✓ (step offsets repeat) | ✓ (new points, same connectivity offsets) | ✓ (new points + connectivity) | ✓ (PartitionedDataSetCollection blocks, each with its own `Steps`) | ParaView ≥ 5.12, VTK 9.6 (`vtkHDFReader`), pyvista, h5py, `vcell-jhdf` | The strongest standard; expresses the piecewise-topology model natively. DOLFINx 0.10's `dolfinx.io.vtkhdf` handles only a static mesh with a single array `u`, so a custom writer is needed. Same HDF5 web and mid-run concerns as XDMF |
| **ADIOS2 BP / VTX** | ✓ | ✓ (`VTXMeshPolicy.update`) | ✗ (a new file per segment) | ✗ | ParaView (ADIOS2 reader) | Best parallel I/O and arbitrary order, but no Java or browser reader |
| **VTU series + PVD** | the mesh is repeated every step | ✓ | ✓ | ✗ | everything, including `VtuGridParser` | Trivial, but storage and bandwidth grow with the mesh times the step count |
| **VTU mesh + zarr fields** | ✓ | ✓ (per-segment coordinates array) | ✓ (segments) | ✓ | Python `zarr`; small Java/JS readers (JSON metadata + zlib chunks) | Our own schema; ParaView can't join the mesh and fields without an export |
| **Exodus II** | ✓ | ✓ (displacement variables) | ✗ (file per segment) | ✓ (element blocks) | ParaView, netCDF-Java | Not DOLFINx-native; heavy writer; same concurrency concerns |

## Decision

1. **Fixed meshes (the common case): VTU mesh + zarr fields.**
   - Each domain's mesh is written **once** as a VTK XML UnstructuredGrid.
   - Each variable is a zarr (format v2) array of shape `(T, N)` of P1 point data, whose columns
     follow the VTU point order.
   - Per-time statistics are a `(T, 4)` array.
   - One self-describing manifest covers the whole bundle.
2. **Moving / remeshed meshes: the same format, extended with segments.**
   - A remesh starts a new segment with its own VTU and arrays.
   - ALE point motion is a per-segment coordinates array.
   - A fixed-mesh run is the one-segment case.
   - The schema reserves this now; it is implemented when the ALE/remesh drivers are wired to the CLI.
3. **ParaView is served by export, not natively.** An export command writes a PVD series or an
   XDMF time series from a bundle. XDMF stays useful as an interchange format, not as the stored
   format.

Why this over VTKHDF, the strongest standard:
- The common case gets the simplest possible layout.
- Every chunk is one addressable object — one file per (variable, time) — so it can be served
  thinly over HTTP and read safely mid-run.
- It continues pyvcell's zarr conventions.
- The Java and JavaScript readers are a few dozen lines.
- The one thing given up, native ParaView opening, is recovered cheaply by export.

## 3. Schema 1

```
<out>/SimID_<key>_<job>_.fenics/        zarr v2 group directory
  .zattrs     {"vcell_fenics": manifest}   written atomically by the writer (temp file + os.replace)
  mesh/<domain>.vtu                        UnstructuredGrid: one Piece, inline binary, no compressor,
                                           UInt32 header, little-endian (VtuGridParser-compatible)
  <domain>/<var>                           zarr (T, N) '<f8', chunks (1, N), zlib, fill NaN
  stats/<domain>/<var>                     zarr (T, 4) columns = stats_columns
  provenance/…, summary.json               convenience copies; not part of the contract
```

**Manifest** (`.zattrs["vcell_fenics"]`):

| key | meaning |
|---|---|
| `schema` | integer, `1` |
| `profile` | `"fixed"` or `"segmented"` |
| `status` | `"running"`, `"completed"` or `"failed"`; `message` holds the failure text |
| `progress` | fraction 0–1 |
| `times` | times of the rows written so far — **authoritative** |
| `planned_times` | all output times the run intends to write |
| `segments` | list of `{index, t0, count, motion, prefix}` |
| `domains` | `{name: {kind: "volume"\|"membrane", dim, gdim, mesh, n_points, n_cells, cell_type}}` |
| `variables` | list of `{name, domain, assoc: "point", element: "P1", path, stats}` |
| `stats_columns` | `["mean", "total", "min", "max"]` |
| `solver` | `{version, dolfinx, options, overrides, mpi_ranks}` |
| `source` | `{kind, file, sim_key, job_index, task_id}` |
| `updated` | ISO-8601 timestamp |

- **Domain names** are the VCell math's CompartmentSubDomain / MembraneSubDomain names, so viewer
  and VCell vocabulary agree.
- **Point order** is canonical and independent of the MPI rank count: owned P1 dofs are keyed by
  `mesh.geometry.input_global_indices` and gathered to rank 0, and cells are sorted
  lexicographically.
- **Segments.** A fixed run has exactly
  `segments = [{index: 0, t0: 0, count: T, motion: "none", prefix: ""}]`. A segmented run adds
  entries with `prefix: "seg0001/"`, each with its own `<prefix>mesh/<domain>.vtu` and
  `<prefix><domain>/<var>`. `motion: "ale"` means `<prefix><domain>/_coords` exists with shape
  `(T_seg, N, 3)`. A reader that follows *segment → prefix → row* handles both profiles the same way.
- **Reader rules:**
  - Refuse a `schema` newer than you know.
  - Ignore unknown keys.
  - Take row counts from `times` (and per-segment `count`), never from array shapes (arrays may be
    preallocated to `planned_times`).
- **Write order:** a row is written before its time is appended to `times`, so a reader polling
  mid-run never sees a row whose data is incomplete.

## Consequences

- vcell-fenics gains a `results/` package (writer, reader, P1 gather, VTU I/O, export) and a
  `zarr` dependency. XDMF is no longer written by default.
- The VCell field viewer needs **point-data** support, and `VtuGridParser` needs `VTK_LINE` (2D
  membranes) and a 3D triangle area. These are recorded in the ADR 011 follow-up.
- pyvcell can read a bundle with a small `FenicsResult` alongside its finite-volume zarr reader.
- `(1, N)` chunks mean T files per variable. That's fine for tens to hundreds of outputs; revisit
  (rows per chunk, or zarr v3 sharding) if runs emit thousands of outputs to NFS.

## 6. Validated by spike

*Pending — step 2 of the integration plan.* To record:
- how zarr-python 3 spells `zarr_format=2` + zlib;
- the stdlib decode round-trip;
- pyvcell's zarr 2.18 reading the bundle;
- a `VtuGridParser` port accepting the VTU;
- `TS.interpolate` output-time accuracy;
- MPI point-order parity.
