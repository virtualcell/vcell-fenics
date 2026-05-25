# vcell-fenics — overview

A short tour for someone new to the codebase. For the deep design, read
[`docs/modeling/declarative-formalism.md`](modeling/declarative-formalism.md);
for the code structure, [`docs/architecture.md`](architecture.md); to see it run,
the notebooks in [`docs/notebooks/`](notebooks/).

## What this is

`vcell-fenics` explores **FEniCSx (DOLFINx) as a backend for cell-mechanics and
cell-migration modelling**, in the context of Virtual Cell. Rather than writing a
finite-element program per model, you write a **declarative description of the
math** and a backend turns it into a solve. The aim is "one math description,
many solvers" — the same model could run on the DOLFINx backend here, or later a
finite-volume backend, or VCell's existing moving-boundary solver.

The first concrete goal — a 2-D cell with two receptor species on a
diffusing, reacting, radially-expanding membrane — runs end-to-end today
(notebook [`02`](notebooks/02_moving_membrane_two_species.ipynb)).

## The three-artifact model

A run is a triple, each referenced by *name* (not file path):

| Artifact | What it is | Where |
|---|---|---|
| **MathDescription** | *What* is solved: subdomains, variables, equations, parameters, BCs, motion. | `vcell_fenics.formalism` |
| **Geometry** | *Where*: the mesh + which regions are which subdomain class. | `vcell_fenics.backend.geometry` (v1 adapter) |
| **SolverConfiguration** | *How approximately*: time step, final time, FE degree. | `vcell_fenics.backend.solver` |

The split means the same MathDescription runs on a test disk or a real cell mesh
unchanged, and a parameter sweep varies only the SolverConfiguration.

## Running a model

```python
from vcell_fenics.formalism import load_yaml
from vcell_fenics.backend import run, SolverConfiguration, make_disk_membrane_geometry

md = load_yaml(MODEL_YAML)                       # parse + (later) validate
geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane")
problem = run(md, geometry, SolverConfiguration(dt=0.01, t_final=1.0))
problem.unknown                                  # the final field(s)
```

A MathDescription is plain YAML (or JSON, or in-memory dataclasses):

```yaml
math_description:
  geometry: disk_membrane
  subdomains:
    - { name: membrane, kind: surface, motion: { kind: prescribed, velocity: "r_dot * x / r(x)" } }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.05" }
      initial_condition: "1.0 + 0.3 * cos(2 * theta(x))"
  parameters:
    - { name: r_dot, value: 1.0 }
```

## What v1 implements (end-to-end, through the formalism)

- **Parser + validator** (`formalism/`): YAML/JSON ↔ dataclasses, a hand-written
  expression parser → typed AST, and a structural validation pass (most of §1.11:
  name resolution, types, coverage, temporality, operator-usage rules).
- **DOLFINx backend** (`backend/`): an expression→UFL compiler, a `DiscreteProblem`
  IR with backward-Euler lowering, a geometry adapter, and the `assemble` / `run`
  driver.
- **Physics**: templates **T1** (bulk reaction-advection-diffusion) and **T2**
  (surface PDE with dilution); `diffusion` and `source` slots; **coupled
  multi-species** systems (one solve over a vector space); **prescribed-velocity
  motion** with **automatic stretch dilution** `ρ ∇_Γ·v_Γ` and a mesh-quality
  guard; scalar variables, constant/expression parameters, spatially-varying ICs;
  zero-Neumann external BCs. Backward Euler, P1 Lagrange, direct LU.

Anything outside that subset is **rejected with a clear error**, never solved
silently wrong.

## Repo layout

```
src/vcell_fenics/
  formalism/        pure-Python: schema, loader/dumper, expr + parser, validator, templates, vocabulary
  backend/          DOLFINx: compiler, discrete (IR + lowering + motion), geometry, assemble, solver
  approaches/       mesh builders (disk, disk+membrane submesh) the backend wraps
  viz.py            PyVista / XDMF helpers
tests/              pytest; test_backend_* drive the conformance models through the formalism
docs/
  modeling/declarative-formalism.md   the design spec (Parts 1–3)
  decisions/                          ADRs (001–005)
  notebooks/                          runnable demos
  overview.md, architecture.md
```

A key property: **`formalism/` has no FEniCSx dependency** (it's pure parsing and
validation); DOLFINx enters only in `backend/`.

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

- **Design rationale**: `docs/modeling/declarative-formalism.md` (the spec) and
  `docs/decisions/` (ADR 004 = the DiscreteProblem IR; ADR 005 = the typing
  posture).
- **Code structure**: `docs/architecture.md`.
- **See it run**: `docs/notebooks/` (01 surface diffusion, 02 the two-species
  moving membrane, 03 validation + the IR).
- **What's deferred** (v2, demand-driven): BCs beyond zero-Neumann and bulk↔surface
  `trace` coupling; the weak-form escape hatch and unknown (mechanics-driven)
  motion; templates T3/T4/T5–T7; remeshing; the full SolverConfiguration.
