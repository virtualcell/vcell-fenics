# FEniCSx coverage of saved VCell BioModels

**Question:** of the spatial models VCell users have saved and run with the finite-volume solvers,
how many can the FEniCSx solver run today, and what blocks the rest?

**Answer (2026-09-28, VCell 8.2.0.04, vcell-fenics `sha-7790431`):** **193 of 600 applications run
(32 %)**, covering 673 of their 2,588 saved simulations. The first survey, two days earlier, found 51 (8 %).
- **Size and speed is now the largest gap** (158 applications, 26 %): 3D runs past the survey's short
  timeouts, over its memory cap, or with a mesh too large at the simulation's own resolution. It's a matter of
  scale, not of a missing feature.
- **FastSystem** models (116, 19 %) are refused by decision.
- **VCell's own math generation fails** for another 67 (11 %).
- **The feature gaps left are small** and add up to about 60 applications. A single solver bug accounts for 11
  of them.

The full generated reports are next to this page:
- [`report-8.2.0.04.md`](coverage-survey/report-8.2.0.04.md), the current one;
- [`report-regenerated-math.md`](coverage-survey/report-regenerated-math.md), the first survey on 2026-09-26;
- [`report-saved-math.md`](coverage-survey/report-saved-math.md), the same first sample on the math as saved.

## Progress

| date | code | ran | what changed |
|---|---|---|---|
| 2026-09-26 | vcell-fenics `8360060`-era | 51 (8 %) | the first survey (below) |
| 2026-09-27 | + #185, #186, #188, #196, #199 | 101 (17 %) | non-diffusing species, region variables and sizes, one-sided membranes |
| 2026-09-28 | + #204, #206 | 154 (26 %) | subvolumes touching the box; the multi-compartment solver (any number of compartments and membranes) |
| 2026-09-28 | + #207 (8.2.0.04) | **193 (32 %)** | single-compartment box-face BCs |

## Method

- **Corpus:** 2,510 public BioModels (`vcml_biomodels/`, downloaded by
  `scripts/download_public_biomodels.py`, gitignored) hold 31,996 simulations. **15,117 simulations in
  3,423 applications** are spatial, deterministic and saved with a finite-volume-family solver
  (Sundials PDE, Finite Volume Regular Grid / Standalone, Chombo, MovingB). Those are the candidates.
- **Sample:** a seeded random sample of **600 applications** (seed 20260926), about 1 in 5.7. The
  percentages estimate the corpus.
- **Math regeneration:** each BioModel is regenerated first (`scripts/regenerate_vcml.py`, libvcell's
  `vcml_to_vcml`), as VCell does before it builds a SimulationTask. The saved math of very old
  documents defines each volume variable in every compartment and each membrane variable in every
  membrane, with no domain. Solvers never see that math, and reading it directly made "legacy
  domain-less variables" look like the largest blocker (203 of the 600) when it isn't one.
- **Run:** each application's first finite-volume simulation goes through the real CLI:
  - `python -m vcell_fenics.cli --vcml … --application …` with the method of lines;
  - the simulation's own mesh size (h = min over axes of extent/mesh size);
  - a short horizon of 3 output intervals;
  - a 300 s timeout (120 s in 3D) and a 6 GB resident-memory cap, one process per application.

  A run that times out after reaching its first output is retried once over a single interval. "Ran"
  means the CLI exited 0 with a results bundle.
- **What it doesn't measure:** accuracy (see the FRAP check below, and `cross_validation/` for the
  standing comparisons), or runs longer than the timeouts. VCell users do run jobs for hours, so the
  timeouts mark runs that aren't interactive-scale, not runs that can't finish.

## Results (2026-09-28)

**193 of 600 applications run (32 %), covering 673 of their 2,588 saved simulations.**

| | applications | ran |
|---|---|---|
| 2D | 333 | 148 (44 %) |
| 3D | 256 | 45 (18 %) |
| 1D | 11 | 0 |
| analytic geometry | 390 | 106 (27 %) |
| image geometry | 199 | 87 (44 %) |
| saved with Sundials PDE | 348 | 109 (31 %) |
| saved with Finite Volume (Regular Grid) | 174 | 62 (36 %) |
| saved with Chombo | 17 | 12 (71 %) |
| saved with MovingB | 5 | 3 (60 %) |

In 3D, 93 of the 256 applications don't finish within the survey's limits (below); 2D runs at 44 %.

### What blocks the rest, and what to do next

Ranked by applications in the sample.

| # | blocker | apps | what it takes |
|---|---|---|---|
| 1 | **3D size and speed:** a timeout (97), a mesh too large at the simulation's own mesh (39), or the 6 GB cap (22) | 158 | #192. MPI (the solver runs under `mpiexec` already; the survey doesn't use it), a mesh-size policy (a body-fitted P1 mesh rarely needs the finite-volume grid's resolution), and solver tuning (reuse the preconditioner across steps). Many would finish as they are on the cluster: the survey's 120 s is an interactive limit. |
| 2 | **FastSystem** (VCell's rapid-equilibrium reduction, e.g. fast calcium buffering) | 116 | Refused by decision (pyvcell drops the FastSystem, so FEniCSx reads it from the XML and refuses). The largest single feature ceiling left. |
| 3 | **VCell cannot generate the math** for the BioModel today (not a FEniCSx gap) | 67 | VCell side. One failing application sinks the whole BioModel file in `vcml_to_vcml`, so some are recoverable by regenerating per application. |
| 4 | **Found multiple domains** (a UFL form built over more than one mesh where one is expected) | 11 | A solver bug, all but a few in one BioModel (98123025): likely a quick fix. |
| 5 | **Math and expression gaps:** formalism validation (11), unresolved names (6), a predicate mixing booleans and numbers (5) | 22 | Small import fixes; each a handful of applications. |
| 6 | **1D geometry** | 11 | Not implemented. A 1D realization (an interval mesh) would be simple, but these are few. |
| 7 | Smaller: CSG subvolumes (4, virtualcell/pyvcell#59), crashes (4), realization failures (3), other (11) | 22 | Case by case. |

Two further findings:
- **VCell's gate offers FEniCSx for 578 of the 600, and 385 of those fail** (virtualcell/vcell#2112).
  Adding the FastSystem check to the Java gate alone would turn 116 of them into up-front refusals.
- **A run's CPU on macOS:** MPICH's default libfabric provider there (`sockets`) spins two threads per
  process for its whole life, so a one-rank solve used about 300 % CPU, and parallel runs stalled a 16-core
  machine. `FI_PROVIDER=tcp` (set by the pixi environment on macOS since vcell-fenics #205) fixes it. It was
  the unexplained "parallel runs thrash" of the first survey. Linux doesn't spin.

### Sanity check

FRAP (BioModel 105875696) was run to completion on both solvers. Its disk has r = 10 and a 10 × 10
bleached square, with dye at 10 everywhere else.
- **Fields:** agree with fvsolver to about 2 % (relative L2 and L∞ at the FV grid points) at every
  output time after t = 0.
- **Recovery plateau:** the exact value is 10·(π·10² − 10²)/(π·10²) = 6.817. FEniCSx gives 6.853
  (+0.5 %) and fvsolver 6.728 (−1.3 %). Each is off only by how its mesh or grid resolves the step
  initial condition and the disk.

A multi-compartment check (nucleus | cytosol | outside, against fvsolver, 0.11 % / 0.05 % / 0.001 % at the
finest grid) is in `cross_validation/README.md`.

## The first survey (2026-09-26)

**51 of 600 applications ran (8 %), covering 135 of their 2,588 saved simulations.** On the saved (stale) math
the figure was 29 (5 %). The blockers then, each filed as an issue:

| # | blocker | apps | issue | now |
|---|---|---|---|---|
| 1 | Membrane species facing a compartment with no species | 129 | #185 | fixed (#203) |
| 2 | Non-diffusing (ODE) species on spatial domains | 118 | #186 | fixed (#202) |
| 3 | A subvolume that touches the domain box | 72 | #187 | fixed (#204) |
| 4 | VCell's own math generation fails | 67 | — | unchanged (VCell side) |
| 5 | 3D too large or too slow at the simulation's own mesh | 69 | #192 | open; now the largest gap |
| 6 | Region variables: membrane potential, region averages | 29 | #188 | fixed (#188, #196) |
| 7 | Volume-scoped functions used on the adjacent membrane | 11 | #189 | fixed (#201) |
| 8 | Field data (`vcField`) | 10 | #190 | open |
| 9 | Membrane-scoped functions unresolved at compile time | 10 | #191 | open (part of row 5 above) |
| 10 | Smaller gaps and crashes (`sim.t`, Netgen 2D, per-face Dirichlet, N compartments, 1D) | ~25 | #193 | mostly fixed (#202, #206, #207) |
| 11 | CSG geometry (pyvcell reads `CSGObject` subvolumes as expression-less analytic ones) | 4 | virtualcell/pyvcell#59 | open |

Several blockers the first survey couldn't see sat behind these, because it records only each application's
first failure. Three or more subvolumes (133 applications) and FastSystem (116) were the largest.

## Reproduce

```bash
.pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py census                      # → coverage/census.csv
../pyvcell/.venv/bin/python scripts/regenerate_vcml.py --sample 600                    # → vcml_biomodels/regenerated/
mkdir -p vcml_biomodels/coverage-regen && cp vcml_biomodels/coverage/census.csv vcml_biomodels/coverage-regen/
pixi run -e dev python scripts/survey_fenics_coverage.py run --vcml-dir vcml_biomodels/regenerated \
    --work vcml_biomodels/coverage-regen --jobs 6 --timeout 300 --timeout-3d 120 --max-rss-gb 6 --sample 600
pixi run -e dev python scripts/survey_fenics_coverage.py report --work vcml_biomodels/coverage-regen --sample 600
```

- **Time:** the sample takes about 2 hours at 6 jobs on a 16-core laptop.
- **CPU:** run it through `pixi run` (or with `FI_PROVIDER=tcp`) on macOS, or each process spins two cores
  (above).
- **Parallel timings:** a run near its timeout can tip over when six run at once, so recheck near-limit
  timeouts one at a time before calling them regressions.
- **Caches:** give each job its own compile cache (the script does). Don't seed them by copying one cache:
  a copied lock file made JIT compilation time out.
- **Resuming:** `runs.csv` is resumable, so a stopped run picks up where it left off.
