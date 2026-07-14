---
name: cross-validate
description: >
  Numerically cross-validate vcell-fenics against pyvcell's fvsolver (fixed-grid finite volume) or
  mbsolver (moving boundary) on a common VCML both consume — checking accuracy (relative-L2), mass
  conservation, and h/dt convergence. Use when asked to compare/validate the FEniCSx backend against
  VCell's solvers, add a new comparison case (2D/3D, diffusion/coupled/membrane/receptor/advection/
  moving-boundary), or extend the `cross_validation/` harness. Two environments are involved.
---

# Cross-validating vcell-fenics ↔ pyvcell (fvsolver / mbsolver)

The comparison runs a **common authored VCML** through VCell's solver (ground truth) and through
`vcell_fenics` (import → realize → solve), then compares on the VCell solver's own grid. It lives in
`cross_validation/`. **Two solvers, two baselines:**

- **fvsolver** — fixed-grid finite volume, implicit surface. Baseline for our **fixed-geometry**
  body-fitted / cut-FEM solves (`run`, `integrate_interface_coupled`, `integrate_membrane_coupled`).
- **mbsolver** — moving front. Baseline for our **ALE moving-boundary** solves (`ale.py`).

## Two environments — this is the core constraint

libvcell + the native solver stay OUT of the DOLFINx env, so the flow is **two stages in two envs**,
bridged by files on disk (a `.npz` reference field + `_math.yaml`/`_geom.yaml` lowered model):

| stage | env | does |
|---|---|---|
| 1 — author + run FV/MB | `../pyvcell/.venv/bin/python` | author VCML, run the solver, dump reference `.npz` + lowered `_math.yaml`/`_geom.yaml` |
| 2 — import + solve + compare | `.pixi/envs/dev/bin/python` | import the yaml (pyvcell_bridge), realize, solve, sample on the FV grid, report metrics |

Never import pyvcell-native and dolfinx in one process. The **yaml is the cross-env bridge** (VCML→
MathDescription lowering needs libvcell, which only exists in the pyvcell env).

## Stage 1 idiom (pyvcell `[native,solver]` env)

```python
import pyvcell.vcml as vc
from pyvcell.vcml.models import Biomodel, Model
from pyvcell.vcml.utils import to_vcml_str
from pyvcell.vcml.vcml_reader import VcmlReader

geo = vc.Geometry(name="box", dim=3, extent=(4.,4.,4.), origin=(-2.,-2.,-2.))  # dim=2 or 3
geo.add_background("domain")                       # or geo.add_analytic_subvolume(...) for a cell
model = Model(name="m"); model.add_compartment("cell", dim=3); model.add_species("u", "cell")
# reactions: model.add_reaction_mass_action(...); general kinetics also available
bm = Biomodel(name="M", model=model)
app = bm.add_application("app", geometry=geo)
app.map_compartment("cell", "domain")
app.map_species("u", init_conc="<vcell expr in x,y,z>", diff_coef=0.1)   # advection: velocity on the species
app.add_sim(name="sim", duration=1.0, output_time_step=0.1, mesh_size=(nx, ny, nz))  # 3D mesh

result = vc.simulate(vc.load_vcml_str(to_vcml_str(bm)), "sim")           # runs fvsolver
# field: result.zarr_dataset is (T, C_all, Z, Y, X); map channels via result.channel_data labels,
# coords x/y/z from the 'x'/'y'/'z' channels, times result.time_points, names result.volume_variable_names.
# (see author_and_run.result_to_4d for the 2D extractor — add z for 3D -> a (T,C,Z,Y,X) field.)

# Dump the lowered model the dev side imports (mesh size is a sim setting, not part of the math):
app2 = VcmlReader.biomodel_from_str(to_vcml_str(bm)).applications[0]
(cv/"stem_geom.yaml").write_text(yaml.safe_dump(app2.geometry.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False))
(cv/"stem_math.yaml").write_text(yaml.safe_dump(app2.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False))
np.savez_compressed(cv/"stem_reference.npz", field=field, t=t, x=x, y=y, z=z, channels=channels)  # gitignore big .npz
result.cleanup()
```

mbsolver stage 1: see `mb_translation_fv.py` (front polygon + inside scatter per frame). Match the
velocity convention carefully — with `v = v_b` (Lagrangian, "cell carries its cytoplasm") the moving-
frame PDE is pure diffusion and the Rankine–Hugoniot BC reduces to no-flux; that matches ALE
`MotionPrescribedVelocity`. Setting only the front velocity is a *different* problem.

## Stage 2 idiom (vcell-fenics dev env)

```python
import pyvcell.vcml.models_geometry as gmod, pyvcell.vcml.models_math as mmod, yaml
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description
from vcell_fenics.backend.realize import realize            # or realize_interface_coupled
gd = import_geometry(gmod.Geometry.model_validate(yaml.safe_load(...geom.yaml)))
md = import_math_description(mmod.MathDescription.model_validate(yaml.safe_load(...math.yaml)), geometry=gd.name, dim=3)
geometry = realize(gd, h=...)                                # 3D realize now supported (ADR 008 §8)
# solve: run(md, geometry) | integrate_interface_coupled(md, geometry, t_final) | integrate_membrane_coupled(...)
# sample u_h at the FV grid points (DOLFINx collision queries — see compare_fenics_vs_fv._eval_on_grid,
# generalize the meshgrid to 3D), then report:
#   relative-L2 per output time; total mass ∫u dV over time (conservation); h/dt convergence rate.
```

## VCell fvsolver mesh & output grid — know this before comparing

`mesh_size=(nx, ny, nz)` sets the FV grid. The `Result`/zarr output then has **nx points per axis
spanning the FULL domain `[origin, origin+extent]` *inclusive***, spacing `dx = extent / (nx − 1)` — so
`x[0] = origin` and `x[-1] = origin+extent` sit **on the domain boundary** (e.g. `mesh_size=32` on
`[-1,1]` → 32 points, `dx = 2/31`, the endpoints on ±1 — *not* cell centers at ±0.969). Read positions
from the `x`/`y`/`z` channels (`zd[0, idx['x'], 0, 0, :]`, etc.).

VCell's FV is **cell-centered, but the DOFs live on the border**: each grid point owns a control volume
centered on it, and the boundary points own **fractional** control volumes — **½ on faces, ¼ on edges,
⅛ on corners** — so the per-DOF control volumes still sum to the exact domain volume. Consequences:

- **Integrating the FV field** (mass/totals): use **trapezoidal** weights (`np.trapezoid` per axis) — that
  *is* the ½/¼/⅛ boundary weighting. A uniform `field.sum()*dx**n` rectangular sum over-counts the boundary
  DOFs and shows a spurious drift as the field spreads to the wall (the "+4 % FV mass" red herring — with
  trapezoidal, FV conserves to round-off). Prefer computing the **FEM** mass as `assemble_scalar(u*dx)` and
  only ever trapezoid-integrating the FV grid data.
- **Sampling for the field comparison** is unaffected: evaluate the FEM at these grid points and compare
  pointwise (relL2/relL∞); the boundary points lie on the FEniCSx box faces, so they resolve.
- **Free-space analytic** only valid while an IC/feature stays ≳ 4–5σ from every boundary — the domain is
  bounded and no-flux by default, so wall reflection (a real physical difference) contaminates it later.

## Membrane binding — the KMOLE-reconciled conserved total

A membrane surface species `R` (density, molecules·µm⁻²) and a volume ligand `s` (µM) live in different
units, so a raw `total_mass()` that sums `∫s dV + ∫R dA` is **not** conserved (it read a spurious ~27×).
The physically conserved substance reconciles them through VCell's **KMOLE** (≈ `1/602.214`, the
molecules↔µmol constant carried in the imported math as a parameter):

```
total_substance = ∫s_cyto dV + ∫s_ext dV + KMOLE · ∫R dA     # all in one substance unit
```

so the free-ligand lost equals `KMOLE · ∫R dA` gained (verified exactly). Pull KMOLE from the imported
parameters — `kmole = next(p.value for p in md.parameters if p.name == "KMOLE")` (≈ 1.6605e-3) — not
hard-coded, then compute `result.mass('s_cyto') + result.mass('s_ext') + kmole*result.mass('R')`
(`mass('R')` is the membrane integral `∫R dA`). Compared to the same total at t=0 (via
`assemble_membrane_coupled(...)`), it is **flat to round-off (~1e-14) at every h** — see
`compare_membrane_3d.py`. (The depleted-ligand match vs the conservative FV validates the binding
regardless; this KMOLE total is the direct conservation check.)

  **Caveat — fine now, not general.** VCell's unit system lives in the *biological* model, not the
  generated (unit-stripped) MathDescription, so trusting `KMOLE` by name/value as the membrane↔volume
  factor is only valid under the **default unit system** — which is almost always the case, so it's OK to
  rely on it (or hard-code `1/602.214`) for now. The general, unit-safe source is the **math-symbol-mapping**
  (math symbols → biological variables *with* units), which pyvcell does not currently persist. If a model
  uses non-default units this can be wrong; revisit if/when vcell-fenics carries optional unit-system
  metadata (populated from the math-symbol-mapping). See `project-vcell-units-in-biology` in memory.

## Templates to copy (nearest case → adapt)

- diffusion (single species): `author_and_run.py` + `compare_fenics_vs_fv.py` (+ `convergence_study.py`)
- interface-coupled (two bulks + permeability): `coupled_perm_fv.py` + `coupled_perm_convergence.py`
- membrane flux / jump condition: `membrane_flux_sign.py`, `membrane_timeflux_fv.py` + `compare_membrane_timeflux.py`
- membrane receptor (surface species): `receptor_fv.py` + `receptor_convergence.py`
- moving boundary (mbsolver ↔ ALE): `mb_translation_fv.py` + `mb_translation.py`

To add a **3D** case: copy the nearest 2D stage-1, switch `vc.Geometry(dim=3)` + `mesh_size=(nx,ny,nz)`,
extend the field extractor to `(T,C,Z,Y,X)`, and on the dev side use the 3D `realize` / 3D coupled
solvers (already validated — see `tests/test_backend_coupled_3d.py`).

## Metrics & gotchas

- **Accuracy:** report **both relative-L2 and relative-L∞** of `u_h` vs the FV field at the FV grid
  points, per output time — L∞ (`max|u_h − fv| / max|fv|`) exposes localized peak/front error that the
  L2 norm averages away (e.g. a sharp under-resolved IC where L∞ ran ~1.5× the L2). Always show both.
  Use the method-of-lines integrator (PETSc TS adaptive BDF) so the time error ≈ 0 and the comparison
  isolates the spatial discretisation (matches FV's Sundials/CVODE). Backward-Euler adds a *our*-side
  dt error.
- **Mass conservation:** `∫u dV` per compartment over time; FV is locally conservative, FEM weakly —
  the interesting axis. Expect exact conservation on closed (no-flux) systems. **Compute the FEM mass as
  `assemble_scalar(u*dx)` on the mesh, NOT a rectangular sum over the FV grid** — the FV output DOFs are
  cell-centered but live *on* the domain boundary with **fractional control volumes** (½ face, ¼ edge, ⅛
  corner), so `field.sum()*dx**n` over-counts the boundary and shows a spurious "drift" as the field
  spreads there (a real +4 % red herring). Integrate the FV field with **trapezoidal** weights
  (`np.trapezoid` per axis) — that *is* the fractional boundary weighting — and FV conserves to round-off.
- **Membrane-species conservation is unit-mixed:** a membrane surface species (density, molecules·µm⁻²)
  and volume species (µM) reconcile only through VCell's KMOLE factor, so a raw `total_mass()` that sums
  them is not the conserved quantity. Validate membrane binding by the **depleted bulk-ligand field match**
  vs the conservative FV (as `receptor_*`/`compare_membrane_3d` do), not a raw total.
- **Convergence:** refine `h` (and the FV `mesh_size`) and check the rate; where an analytic solution
  exists, compare to it (this repo requires h/dt-refinement against analytical, not just discrimination).
- Body-fitted realize + coupled MOL block solves cap at coarser `h` than FV — keep FEniCSx grids modest.
- Big `.npz` references are **gitignored and regenerable** — commit the `.vcml` + `_math.yaml`/`_geom.yaml`,
  not the fields.
- `cross_validation/README.md` documents the committed cases and results — update it when adding one.

The full context of *why* (fvsolver ↔ fixed-grid, mbsolver ↔ ALE) is in the memory
`reference_vcell_geometry_pipeline` and `project_fvsolver_comparison`.
