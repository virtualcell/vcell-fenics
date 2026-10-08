# vcell-fenics — overview

> **Documentation review (2026-10-08):** See the [documentation map](README.md) and
> [discrepancy register](reviews/2026-10-08-documentation-discrepancies.md) for distinctions
> between historical plans and current implementation. New modeling discussion lives in the
> [cell-mechanics workspace](modeling/cell-mechanics/README.md).

A short tour for someone new to the codebase. For the deep design, read
[`docs/modeling/declarative-formalism.md`](modeling/declarative-formalism.md);
for the code structure, [`docs/architecture.md`](architecture.md); to see it run,
the notebooks in [`docs/notebooks/`](notebooks/). The capability list below is dated
**2026-10-08** and names its entry points; the earlier v1 summary it replaces is in this file's
history.

## What this is

`vcell-fenics` explores **FEniCSx (DOLFINx) as a backend for cell-mechanics and
cell-migration modelling**, in the context of Virtual Cell, and runs as a **VCell solver**: a
VCell SimulationTask in, a VTU + zarr results bundle out ([ADR 011](decisions/011-vcell-solver-contract.md),
[ADR 010](decisions/010-results-bundle-vtu-zarr.md)). Rather than writing a finite-element program per
model, you write a **declarative description of the math** and a backend turns it into a solve. The
aim is "one math description, many solvers" — the same model runs on the DOLFINx backend here and on
VCell's finite-volume and moving-boundary solvers, which is how it is cross-validated
([`cross_validation/`](../cross_validation/README.md)).

## The three-artifact model

A run is a triple, each referenced by *name* (not file path):

| Artifact | What it is | Where |
|---|---|---|
| **MathDescription** | *What* is solved: subdomains, variables, equations, parameters, BCs, motion. | `vcell_fenics.formalism` (doc §2) |
| **GeometryDescription** | *Where*: VCell-style subvolumes (analytic / CSG / image / compartmental), surface classes, named faces — the source of truth for the domain ([ADR 007](decisions/007-geometry-source-of-truth.md)). Realized into a body-fitted Netgen mesh by `backend/realize.py`. | `vcell_fenics.formalism.geometry_schema`, [geometric formalism](modeling/geometric-formalism.md) |
| **SolverConfiguration** | *How approximately*: time step, final time, FE degree, time integration (backward Euler or method of lines). | `vcell_fenics.backend.solver` |

The split means the same MathDescription runs on a test disk or a real cell mesh
unchanged, and a parameter sweep varies only the SolverConfiguration.

## Running a model

The one-shot runner is the CLI (`vcell_fenics.cli`; the solve lives in `vcell_fenics.runner`): a VCell
SimulationTask, a `.vcml`, a VCell math + geometry YAML pair, or a native formalism pair in — a results
bundle out. The commands are in [`CLAUDE.md`](../CLAUDE.md); the container is
[`docker/README.md`](../docker/README.md).

In Python, the single-mesh path is three calls:

```python
from vcell_fenics.formalism import load_yaml
from vcell_fenics.backend import run, SolverConfiguration, make_disk_membrane_geometry

md = load_yaml(MODEL_YAML)                       # parse + validate
geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane")
problem = run(md, geometry, SolverConfiguration(dt=0.01, t_final=1.0))
problem.unknown                                  # the final field(s)
```

A MathDescription is plain YAML (or JSON, or in-memory dataclasses). The reserved names are
namespaced ([ADR 006](decisions/006-namespaced-builtins.md)): `geom.x`, `geom.radius`, `geom.azimuth`,
`sim.t`.

```yaml
math_description:
  geometry: disk_membrane
  subdomains:
    - { name: membrane, kind: surface, motion: { kind: prescribed, velocity: "r_dot * geom.x / geom.radius" } }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.05" }
      initial_condition: "1.0 + 0.3 * cos(2 * geom.azimuth)"
  parameters:
    - { name: r_dot, value: 1.0 }
```

## What runs today (2026-10-08)

Three milestones are distinct for every feature: the **formalism** can express it, a **backend
driver** solves it, and the **CLI** routes to that driver. The matrix names the entry point for each.

**Through the CLI / runner** (`src/vcell_fenics/runner.py` picks the path from the model, not a flag):

| Model shape | Path | Notes |
|---|---|---|
| Equations on **one** subdomain, fixed geometry | `realize` → `assemble` → backward Euler (`solver.run`) or method of lines (`reaction_diffusion.integrate_discrete_problem`, PETSc `TS` BDF, GMRES + ILU) | T1 bulk reaction-advection-diffusion, T2 surface PDE with dilution, spatial T4 (a field without transport), coupled multi-species, Dirichlet / Neumann / Robin BCs on labelled faces, time-dependent data through `sim.t`. Nonlinear sources need the method of lines (backward Euler raises `NonlinearTermError` naming that fix); a SimulationTask defaults to the method of lines. |
| Equations on **two or more** subdomains, fixed geometry | `realize_multi_compartment` → `multi_compartment.integrate_multi_compartment` (method of lines, exact or matrix-free Newton) | Any number of compartments and membranes; species on each, **membrane species** (receptor–ligand binding), **region variables** (T5 `region_ode`: a well-mixed species, a membrane potential), box-face values per compartment, analytic and **image** geometries ([ADR 012](decisions/012-image-geometry-realization.md)), 2D and 3D, MPI. |
| A **moving** subdomain (a VCell moving-boundary front) | `realize` → `assemble` → backward Euler through the ALE driver (`ale.run_moving_with_remeshing`) | One prescribed-velocity front, species inside it, 2D and 3D, remeshing when the mesh degrades (2D: Netgen on the deformed polygon with a conservative remap; 3D: `remesh_3d` + interpolation with a global mass correction), P1 only, a new bundle segment per remesh. The velocity may depend on the species (one-step lag). |

Refused with a message, never mis-solved: FastSystem, field data, steady tasks, `StartTime ≠ 0`
(`pyvcell_bridge/simtask.py`); membrane species on a moving front, species outside the moving
compartment, more than one front, image geometries on a moving front; region variables outside the
multi-compartment solver; boundary conditions across a remesh; `region_size` on a moving mesh.

**Backend drivers with their own Python entry points, not routed by the CLI** (each verified by its
tests; the status of each is in the note it implements):

| Driver | What it solves | Where |
|---|---|---|
| `backend/coupled.py` | one bulk + one surface on its boundary, coupled by `trace(·)` (§1.6.6) through `assemble()` | formalism §3.6 |
| `backend/interface_coupled.py` | the older two-compartment and two-compartment-plus-membrane integrators, kept for their tests and cross-validation | formalism §3.6 |
| `backend/weakform.py`, `backend/unknown_motion.py` | the weak-form escape hatch (§1.5) and mechanics-driven membrane motion with curvature forces (§1.10.8) | formalism §3.6 |
| `backend/stokes.py`, `stokes_hdiv.py`, `slip.py`, `multiphase.py`, `fsi.py` | incompressible Stokes (Taylor–Hood and H(div)), Nitsche slip, the two-phase overdamped mixture with drag, and the dynamic FSI loops (prescribed, surface-tension force balance, two-phase) with co-moving species | [multiphase note](modeling/multiphase-cytoplasm-ale.md) §9–10 |
| `backend/cahn_hilliard.py` (`run_cahn_hilliard`, the `cahn_hilliard` template) | resolved diffuse-interface phase separation (condensates), convex-splitting, energy-stable | [approaches](modeling/approaches.md) §C |
| `core/` remaps and `bgn_curve*` | conservative surface and 2D bulk remaps across a remesh, the Approach-A trace correction, BGN tangential redistribution | [surface remap](modeling/conservative-surface-remap.md), [ALE driver](modeling/ale-remesh-driver.md) |

**Verification is standing infrastructure.** [`mms/`](../mms/README.md) is the manufactured-solution
suite with an order-regression gate, run against this backend and VCell's fvsolver / mbsolver;
[`cross_validation/`](../cross_validation/README.md) compares whole models against both VCell solvers.
`tests/` holds the per-driver conformance, conservation, eigenmode and convergence tests.

## Repo layout

```
src/vcell_fenics/
  formalism/        pure Python, no FEniCSx: math schema, loader/dumper, expression parser, validator,
                    templates (T1–T5, cahn_hilliard, weak_form), vocabulary; the geometry formalism
                    (geometry_schema / _io / _validator) and the Rvachev implicit-field lowering
  backend/          DOLFINx: compiler (AST → UFL), DiscreteProblem IR, assemble, realize (Netgen),
                    solver (BE), reaction_diffusion (MOL), multi_compartment, ale + remesh_3d,
                    labels + label_surfaces (image geometries), diagnostics, and the dedicated drivers
                    (coupled, interface_coupled, weakform, unknown_motion, stokes*, slip, multiphase,
                    fsi, cahn_hilliard)
  core/             approach-agnostic NumPy kernels + DOLFINx bridges: surface / bulk remaps, the
                    Netgen region remesher, BGN redistribution
  pyvcell_bridge/   VCell in: SimulationTask XML, VCML/pyvcell math + geometry, expressions, overrides
  results/          the results bundle (ADR 010): writer, reader, recorder, VTU, ParaView export
  runner.py, cli.py, status.py   the one-shot runner, its argv, the VCell status protocol
  viz.py            PyVista / XDMF helpers
tests/              pytest (82 files); test_backend_* drive the conformance models through the formalism
mms/                manufactured-solution suite (README is the source of truth)
cross_validation/   comparisons against VCell's fvsolver / mbsolver (README is the source of truth)
docker/             the container image and its reference
docs/               see README.md (the documentation map)
```

A key property: **`formalism/` has no FEniCSx dependency** (it's pure parsing and
validation); DOLFINx enters only in `backend/`. `src/` is gmsh-free: the mesher is LGPL Netgen
([ADR 008](decisions/008-gmsh-license-isolation.md)); gmsh is a dev/test-only dependency.

## Working in the repo

```bash
pixi install                 # resolve + install (after editing deps)
pixi run -e dev check        # the gate: ruff lint + format-check + mypy --strict + pytest
pixi run -e dev test         # just the tests
pixi run -e dev test tests/test_backend_dilution.py
pixi run -e dev notebooks     # launch JupyterLab on docs/notebooks/
```

`pixi run -e dev check` must stay green. See [`CLAUDE.md`](../CLAUDE.md) for the
quality conventions (e.g. how FEniCSx's partial type stubs are handled — ADR 005).

The notebooks are committed **pre-executed** (plots and outputs render without
running). To run them interactively, `pixi run -e dev notebooks` opens JupyterLab
rooted at `docs/notebooks/`; the dev env's Python is the kernel.

## Where to go next

- **The map**: [`docs/README.md`](README.md) — which document owns what.
- **Design rationale**: `docs/modeling/declarative-formalism.md` (the spec) and
  `docs/decisions/` (ADRs 001–012; ADR 004 = the DiscreteProblem IR; ADR 005 = the typing
  posture; ADR 011 = the VCell solver contract).
- **Code structure**: `docs/architecture.md`.
- **Progress and what's next**: the [integration tracker](integration/vcell-solver-integration.md).
- **See it run**: `docs/notebooks/` (01 surface diffusion, 02 the two-species
  moving membrane, 03 validation + the IR).
- **Still outside the backend** (demand-driven): the mechanics templates T6–T8 (Stokes, elasticity,
  hyperelasticity — the drivers above are not templates), T3 and point-subdomain T4 in the backend,
  the value-equality interface BC, membrane species on a moving front, remeshing a moving 3D
  *surface*, a 3D conservative supermesh remap, BCs across a remesh, the full §3.4
  SolverConfiguration. The formalism's Appendix B is the consolidated list.
