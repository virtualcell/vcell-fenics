# FEniCSx coverage of saved VCell BioModels

**Question:** of the spatial models VCell users have saved and run with the finite-volume solvers,
how many can the FEniCSx solver run today, and what blocks the rest?

**Answer (2026-09-26):** about **8 %** of applications run end to end, and the blockers are a short,
ranked list of specific gaps. The largest four account for about 70 % of the failures. Most of the
failures reach the user as a failed run rather than an up-front refusal, because VCell's pre-run gate
checks only the geometry (virtualcell/vcell#2112).

The full generated reports are next to this page:
[`report-regenerated-math.md`](coverage-survey/report-regenerated-math.md) (the one to read) and
[`report-saved-math.md`](coverage-survey/report-saved-math.md) (the same sample on the math as saved).

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
  - one application at a time, with a 300 s timeout (120 s in 3D) and a 6 GB resident-memory cap.

  A run that times out after reaching its first output is retried once over a single interval. "Ran"
  means the CLI exited 0 with a results bundle.
- **What it doesn't measure:** accuracy (see the FRAP check below, and `cross_validation/` for the
  standing comparisons), or runs longer than the timeouts. VCell users do run jobs for hours, so the
  timeouts mark runs that aren't interactive-scale, not runs that can't finish.

## Results

**51 of 600 applications run (8 %), covering 135 of their 2,588 saved simulations.** On the saved
(stale) math the figure was 29 (5 %).

| | applications | ran |
|---|---|---|
| 2D | 333 | 27 (8 %) |
| 3D | 256 | 24 (9 %) |
| 1D | 11 | 0 |
| analytic geometry | 390 | 38 (10 %) |
| image geometry | 199 | 13 (7 %) |
| saved with Chombo | 17 | 10 (59 %) |
| saved with MovingB | 5 | 3 (60 %) |

### What blocks the rest

Ranked by applications in the sample; each is filed as an issue.

| # | blocker | apps | sims | issue |
|---|---|---|---|---|
| 1 | Membrane species facing a compartment with no species (one-sided membrane coupling) | 129 | 509 | #185 |
| 2 | Non-diffusing (ODE) species on spatial domains (`lumped_ode`, and bulk ODE species in coupled models) | 118 | ~380 | #186 |
| 3 | A subvolume that touches the domain box (2D 38, 3D 34) | 72 | 226 | #187 |
| 4 | VCell's own math generation fails for the BioModel today (not a FEniCSx gap) | 67 | 569 | — |
| 5 | 3D too large or too slow at the simulation's own mesh (timeout 44, mesh too large 23, memory 2) | 69 | ~380 | #192 |
| 6 | Region variables: membrane potential, region averages | 29 | — | #188 |
| 7 | Volume-scoped functions used on the adjacent membrane (§1.11.10 scoping) | 11 | — | #189 |
| 8 | Field data (`vcField`) | 10 | — | #190 |
| 9 | Membrane-scoped functions unresolved at compile time | 10 | 24 | #191 |
| 10 | Smaller gaps and crashes (`KeyError 'sim.t'`, Netgen 2D failures, per-face Dirichlet in coupled models, N compartments, 1D) | ~25 | — | #193 |
| 11 | CSG geometry: pyvcell reads `Type="CSGObject"` subvolumes as expression-less analytic ones | 4 | 21 | virtualcell/pyvcell#59 |

Two further findings:
- **One failing application sinks the whole BioModel file in regeneration** (libvcell's
  `vcml_to_vcml`), so some of row 4's 67 applications may be fine. Row 4 also includes one libvcell
  environment gap (`vcell.installDir` not set).
- **VCell's gate offers FEniCSx for 578 of the 600, and 527 of those fail.** The structural checks for
  rows 1, 2, 6, 8 and 10 are visible in the MathDescription (virtualcell/vcell#2112).

### Sanity check

FRAP (BioModel 105875696) was run to completion on both solvers. Its disk has r = 10 and a 10 × 10
bleached square, with dye at 10 everywhere else.
- **Fields:** agree with fvsolver to about 2 % (relative L2 and L∞ at the FV grid points) at every
  output time after t = 0.
- **Recovery plateau:** the exact value is 10·(π·10² − 10²)/(π·10²) = 6.817. FEniCSx gives 6.853
  (+0.5 %) and fvsolver 6.728 (−1.3 %). Each is off only by how its mesh or grid resolves the step
  initial condition and the disk.

## Reproduce

```bash
.pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py census                      # → coverage/census.csv
../pyvcell/.venv/bin/python scripts/regenerate_vcml.py --sample 600                    # → vcml_biomodels/regenerated/
mkdir -p vcml_biomodels/coverage-regen && cp vcml_biomodels/coverage/census.csv vcml_biomodels/coverage-regen/
.pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py run --vcml-dir vcml_biomodels/regenerated \
    --work vcml_biomodels/coverage-regen --jobs 1 --timeout 300 --timeout-3d 120 --max-rss-gb 6 --sample 600
.pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py report --work vcml_biomodels/coverage-regen --sample 600
```

The sample run takes about 3 hours at one job. Run one job at a time on a laptop: parallel runs on
macOS drove the load average into the hundreds while memory and swap stayed flat, and even `ps`
stalled. The cause wasn't pinned down. `runs.csv` is resumable, so a stopped run picks up where it
left off.
