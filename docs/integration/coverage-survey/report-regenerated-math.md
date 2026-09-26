# FEniCSx coverage of saved VCell biomodels

*A seeded random sample (seed 20260926) of 600 applications; counts below are over the sample, so percentages estimate the whole corpus.*

Corpus: 2510 biomodels, 31996 simulations, of which **15117 simulations in 3423 applications** are spatial, deterministic and saved with a VCell finite-volume-family solver. 600 of those applications were run through the FEniCSx CLI.

**Ran: 51 / 600 applications (8%), covering 135 / 2588 of their simulations.**

## By solver

| solver | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| Sundials Stiff PDE Solver (Variable Time Step) | 348 | 26 (7%) | 40 | 282 |
| Finite Volume, Regular Grid | 174 | 9 (5%) | 3 | 162 |
| Finite Volume Standalone, Regular Grid | 56 | 3 (5%) | 1 | 52 |
| Chombo Standalone | 17 | 10 (59%) | 2 | 5 |
| MovingB | 5 | 3 (60%) | 0 | 2 |

## By dimension

| dimension | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| 2D | 333 | 27 (8%) | 3 | 303 |
| 3D | 256 | 24 (9%) | 43 | 189 |
| 1D | 11 | 0 (0%) | 0 | 11 |

## By geometry

| geometry | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| analytic | 390 | 38 (10%) | 12 | 340 |
| image | 199 | 13 (7%) | 34 | 152 |
| mixed(analytic+csg) | 9 | 0 (0%) | 0 | 9 |
| csg | 2 | 0 (0%) | 0 | 2 |

## By moving boundary

| moving boundary | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| fixed | 596 | 49 (8%) | 46 | 501 |
| moving | 4 | 2 (50%) | 0 | 2 |

## Why applications don't run (by category, ranked)

| category | applications | simulations | what it means |
|---|---|---|---|
| one-sided membrane species | 129 | 509 | membrane species with bulk species in one compartment only (#183) |
| non-diffusing species (lumped_ode) | 93 | 316 | ODE (non-diffusing) species on a spatial domain |
| subvolume touches the box | 72 | 226 | a subvolume boundary touches the domain box (2D/3D realization) |
| VCell cannot generate the math | 67 | 569 | VCell's own math generation fails for this BioModel today (not a FEniCSx gap) |
| math validation | 51 | 264 | the imported math fails formalism validation |
| timeout | 44 | 218 | no result within the timeout at the simulation's own mesh |
| not implemented (other) | 34 | 100 | another feature not implemented yet |
| mesh too large | 23 | 155 | the simulation's own mesh size exceeds the FEM mesh limit (may run coarser) |
| other | 11 | 26 |  |
| unresolved name | 10 | 24 | an expression names something the compiler cannot resolve |
| 1D geometry | 8 | 18 | one-dimensional geometry |
| CSG geometry | 4 | 21 | constructive solid geometry subvolumes (pyvcell reads them as expression-less analytic ones) |
| memory | 2 | 6 | past the survey's per-run memory cap at the simulation's own mesh |
| 3+ compartments | 1 | 1 | species on three or more compartments, or compartments without a membrane between them |

## VCell's gate vs the run

VCell's pre-run check would offer FEniCSx for 578 of 600 applications; **527 of those then fail in the CLI** (a user finds out from a failed run, not an up-front refusal). By category:

| category | offered but fails |
|---|---|
| one-sided membrane species | 129 |
| non-diffusing species (lumped_ode) | 93 |
| VCell cannot generate the math | 66 |
| subvolume touches the box | 63 |
| math validation | 51 |
| timeout | 44 |
| not implemented (other) | 34 |
| mesh too large | 23 |
| other | 11 |
| unresolved name | 10 |
| memory | 2 |
| 3+ compartments | 1 |

## Individual messages (normalized, ranked)

| reason | applications | simulations | example |
|---|---|---|---|
| RunError: membrane species on '…' with bulk species only in '…': the membrane-coupled solver needs species in both compartments it separates ('…' has none) — no | 129 | 509 | 101336098 / Copy of Completebleach |
| NotImplementedError: backend v1 supports templates ['…', '…'], not '…' | 93 | 316 | 10011510 / spatial |
| FormalismValidationError: MathDescription failed validation with # error(s): | 51 | 264 | 120428417 / combined_spatial |
| NotImplementedError: 2D realization requires the boundary of '…' to be strictly inside the box (a subvolume touching the box boundary is a later slice) | 38 | 108 | 10014084 / 1D |
| NotImplementedError: 3D realization requires the boundary of '…' to be strictly inside the box (a subvolume touching the box boundary is a later slice) | 34 | 118 | 101963252 / extra: simple test geometry.  "few fingers" |
| RealizationError: image geometry '…' at h = # would need ~# tetrahedra (the limit is #); use h ≥ # | 23 | 155 | 101232348 / experiment |
| NotImplementedError: membrane coupling needs a bulk diffusion equation for '…' | 19 | 47 | 110218606 / NE_Full_new |
| no regenerated VCML: failed to generate math: generated an invalid mathDescription: Initial condition for variable '…' references variable '…'. Initial conditio | 12 | 240 | 39669796 / Spatial 1 - 15nm shell - PIP2 seq at PSD |
| no regenerated VCML: failed to generate math: exception updating sizes | 12 | 29 | 79151545 /  Spatial fried egg no GDI |
| no regenerated VCML: failed to generate math: '…' is either not found in your model or is not allowed to be used in the current context. Check that you have pro | 10 | 123 | 11006563 / volt&calcium spatial |
| CompileError: unresolved name '…' (not in the compile context) | 10 | 24 | 131370534 / 0.3x0.42-Diff-Ca-0.5-New6th-Spark1-Gain200-sigma0.0099-BC-XValue-YFlux |
| no regenerated VCML: failed to generate math: species '…' interacts with surface '…', but is not mapped spatially adjacent | 9 | 51 | 35512127 / Spatial 1 - 20nm shell |
| no regenerated VCML: failed to generate math: structure cyt not mapped, or unsupported GeometryClass null | 8 | 91 | 125971612 / 2D pde analytic |
| NotImplementedError: realization of dim=# geometry '…' is not implemented yet (supports dim #, #, #) | 8 | 18 | 31628467 / FRAP |
| NotImplementedError: interface coupling needs a bulk diffusion equation for each species on compartment '…' | 6 | 39 | 110197099 / spatial |
| KeyError: '…' | 4 | 16 | 109232682 / timed |
| NotImplementedError: Dirichlet BC on boundary '…': the membrane-coupled solver supports a reservoir Dirichlet only on the outer box boundary '…' | 4 | 7 | 19822057 / astrocyte |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   Size parameter is not set. Size parameter is not set. Size parameter  | 4 | 10 | 34779681 / spatial_image |
| RealizationError: 3D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 3 | 17 | 108026026 / AbsorbableKon |
| NotImplementedError: backend v1 couples equations on a single shared subdomain; got ['…', '…', '…'] (cross-subdomain coupling via trace is a later increment) | 3 | 4 | 2078533 / 3boxes |
| NgException: meshing failed | 3 | 3 | 252674703 / Copy of Application0 |
| NotImplementedError: BC boundary '…' bounds ('…',), not this solve'…'cell' | 2 | 3 | 23612320 / FRAP |
| no regenerated VCML: failed to generate math: Application  endosome1 : H_ion_endosome has spatially resolved flux at membrane endosome, but doesn't diffuse in c | 2 | 3 | 32885648 / endosome12 |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   VCell spatial stochastic models only support 3D geometry.   See the P | 2 | 3 | 50216219 / pde_2D_shape |
| no regenerated VCML: failed to generate math: structure nuc not mapped, or unsupported GeometryClass null | 2 | 2 | 6081851 / Ascaris |
| no regenerated VCML: failed to generate math: Application  Spatial : r_cytosol has spatially resolved flux at membrane cytosol, but doesn't diffuse in compartme | 1 | 1 | 11708030 / Spatial |
| RealizationError: 2D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 1 | 4 | 117544101 / Application0 |
| no regenerated VCML: failed to generate math: required System property '…' not defined | 1 | 7 | 188175029 / Random Noise |
| no regenerated VCML: failed to generate math: org.vcell.util.ConfigurationException: required System property '…' not defined | 1 | 1 | 222669359 / 3D |
| SolveError: the residual became non-finite (NaN/Inf) at t ≈ # — likely a blow-up (e.g. an autocatalytic source growing without bound), a division by a quantity  | 1 | 3 | 31910148 / pde_dumble shape |
| SolveError: the source term is non-finite (NaN/Inf) at the initial condition (t = #), before any step — a division by a quantity that is zero at t=#, or a fract | 1 | 2 | 32158617 / Spatial |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   Clamped Species must be continuous rather than particles. Clamped Spe | 1 | 4 | 55073148 / spatial |
| no regenerated VCML: failed to generate math: structure out not mapped, or unsupported GeometryClass null | 1 | 1 | 6085954 / C_elegans |
| IndexError: list index out of range | 1 | 1 | 61710658 / Application0 |
| no regenerated VCML: both volume and membrane functions must be defined for ConvolutionDataGenerator | 1 | 3 | 78352114 / Spatial fried egg |
| RvachevLoweringError: cannot lower a predicate that mixes boolean and numeric operands under '…' | 1 | 1 | 95142058 / simple_transport_simulation_spatialGeom |
| RunError: geometry '…' has more unmodelled subvolumes than the coupled solver can drop (Cell, RE, background); it supports two compartments plus one background | 1 | 1 | 95496528 / Whole cell |

