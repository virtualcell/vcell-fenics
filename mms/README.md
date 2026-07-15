# Manufactured-solution (MMS) verification suite

A **persistent** suite of Method-of-Manufactured-Solutions cases, each recorded as a **math description**
whose `source` term is the manufactured forcing that makes a chosen exact solution `u*(x, t)` the *true*
solution of the PDE. Each case runs through the solvers and its numerical result is compared to the
**known** `u*` — measuring the true `L2`/`L∞` error and the **convergence order**.

Why MMS (vs the conservation tests): a scheme can conserve every integral exactly while converging at the
wrong rate, or to the wrong answer. MMS checks the *solution itself* against a closed form, so it catches
**correctness** defects — wrong order, stale boundary data, term-consistency bugs — that conservation is
blind to. (The moving-bulk case below was found this way in the first hour of building the suite.)

## Three solvers, one manufactured model

Each case is solver-agnostic — a math description + an exact solution + metadata — so the **same** case is
checked against every applicable solver:

| solver | runner | env | applies to |
|---|---|---|---|
| vcell-fenics (BE + MOL/ALE) | `runner.py` | `.pixi/envs/dev` | static + moving |
| pyvcell **fvsolver** (fixed-grid FV) | `runner_fv.py` *(planned)* | `../pyvcell/.venv` | static geometry |
| pyvcell **mbsolver** (moving front) | `runner_mb.py` *(planned)* | `../pyvcell/.venv` | moving boundary |

`applicable_solvers` in each case declares which apply (fvsolver is fixed-grid, so it skips moving-boundary
cases; mbsolver is for the moving ones). The fv/mb runners author the case's math as VCML (general-kinetics
source = the forcing — VCell authoring supports spatial/time-dependent sources), run the native solver, and
compare to the same `u*` on that solver's own grid — the two-env pattern of the `cross-validate` skill,
here against a *known* solution rather than solver-vs-solver.

## Running

```bash
.pixi/envs/dev/bin/python mms/runner.py --all                 # every case, vcell-fenics
.pixi/envs/dev/bin/python mms/runner.py mms/cases/<case>.yaml # one case
```

## Case format (`cases/*.yaml`)

```yaml
name: ...
description: ...            # what it exercises; note if it exposes a defect
harness: bulk              # runner dispatch: bulk | membrane | coupled
dim: 2
geometry: {kind: disk, radius: 1.0, volume_subdomain: cyto, boundary: edge}
exact: {c: "2 + np.sin(x)*np.cos(y)"}   # u*(x,y,z,t) — numpy-evaluable, for the error norm
t_final: 1.0
dt: 0.01
resolutions_h: [0.1, 0.05, 0.025]
expected_order_h: 2.0
applicable_solvers: [fenics-be, fenics-mol, mbsolver]
params: {k: 0.25, D: 0.1}    # for reference / VCML authoring
math: |                      # the vcell-fenics MathDescription: the manufactured source lives in `source`,
  math_description: ...      # the IC is u*(·,0), the BC is u* on the boundary
```

**The load-bearing discipline — every moving case must be validated against its static reduction.** A
manufactured forcing is only trustworthy if the *static* version of the case (motion off) converges at the
theoretical order; if it doesn't, the forcing derivation (or the runner) is wrong, not the solver. Each
moving case therefore has, or is paired with, its static twin (see `static_bulk_diffusion.yaml`). A moving
case that fails while its static twin passes is a genuine solver finding.

## Manufacturing the forcing

For a lab-frame solution `u*` on a mesh moving (Lagrangian) at velocity `v`, the ALE conservation-form
residual is `∂_t u* + ∇·(u* v) − D∇²u* − reaction(u*)`, so the source is

```
f = ∂_t u* + ∇·(u* v) − D∇²u* − reaction(u*)      (= v·∇u* + u*∇·v + ∂_t u* − D∇²u* − reaction  )
```

A *static* case is the `v = 0` reduction, `f = ∂_t u* − D∇²u* − reaction`. On a moving boundary, a
position-dependent Dirichlet `u*` must be re-evaluated at the moved boundary each step (the runner calls
`set_time` to refresh it — see the finding below).

## Findings so far

- **`moving_bulk_expansion_diffusion` — OPEN DEFECT.** A steady `u*` on an expanding disk is *not* held:
  the `L∞` error does not converge in `h` (order ≈ 0.1 for both BE and MOL) and grows with deformation,
  while the static twin converges at order ≈ 2. Root cause under investigation — two contributions found:
  (1) a moving-mesh **Dirichlet-BC refresh** gap (`step()` moves the mesh but does not re-evaluate a
  position-dependent Dirichlet BC at the moved boundary unless `set_time` is called — worth ~100× of the
  error); (2) a residual `~4e-3` consistency error in the ALE transport of a spatially-varying field that
  persists after the BC is refreshed. Conservation tests pass on this exact setup — MMS is what exposed it.

## Roadmap

- Fill the matrix (solver × motion × physics) — diffusion, advection, reaction, dilution, binding; BE, MOL,
  coupled, membrane, unknown-motion; translation, expansion, shrinkage, shear, non-affine.
- The `runner_fv.py` / `runner_mb.py` two-env runners (author VCML, run native solver, compare to `u*`).
- Promote the passing cases into the `pixi run check` gate (order-regression guard); track the open defects.
