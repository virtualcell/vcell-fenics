# FEniCSx coverage of saved VCell biomodels

*A seeded random sample (seed 20260926) of 600 applications; counts below are over the sample, so percentages estimate the whole corpus.*

Corpus: 2510 biomodels, 31996 simulations, of which **15117 simulations in 3423 applications** are spatial, deterministic and saved with a VCell finite-volume-family solver. 600 of those applications were run through the FEniCSx CLI.

**Ran: 193 / 600 applications (32%), covering 673 / 2588 of their simulations.**

## By solver

| solver | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| Sundials Stiff PDE Solver (Variable Time Step) | 348 | 109 (31%) | 104 | 135 |
| Finite Volume, Regular Grid | 174 | 62 (36%) | 13 | 99 |
| Finite Volume Standalone, Regular Grid | 56 | 7 (12%) | 2 | 47 |
| Chombo Standalone | 17 | 12 (71%) | 0 | 5 |
| MovingB | 5 | 3 (60%) | 0 | 2 |

## By dimension

| dimension | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| 2D | 333 | 148 (44%) | 26 | 159 |
| 3D | 256 | 45 (18%) | 93 | 118 |
| 1D | 11 | 0 (0%) | 0 | 11 |

## By geometry

| geometry | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| analytic | 390 | 106 (27%) | 78 | 206 |
| image | 199 | 87 (44%) | 35 | 77 |
| mixed(analytic+csg) | 9 | 0 (0%) | 6 | 3 |
| csg | 2 | 0 (0%) | 0 | 2 |

## By moving boundary

| moving boundary | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| fixed | 596 | 191 (32%) | 119 | 286 |
| moving | 4 | 2 (50%) | 0 | 2 |

## Why applications don't run (by category, ranked)

| category | applications | simulations | what it means |
|---|---|---|---|
| FastSystem | 116 | 320 | VCell's rapid-equilibrium reduction (fast buffering): refused, not solved (a decision, not a gap) |
| timeout | 97 | 517 | no result within the timeout at the simulation's own mesh |
| VCell cannot generate the math | 67 | 569 | VCell's own math generation fails for this BioModel today (not a FEniCSx gap) |
| mesh too large | 39 | 197 | the simulation's own mesh size exceeds the FEM mesh limit (may run coarser) |
| memory | 22 | 55 | past the survey's per-run memory cap at the simulation's own mesh |
| 1D geometry | 11 | 27 | one-dimensional geometry |
| math validation | 11 | 99 | the imported math fails formalism validation |
| multiple domains (UFL) | 11 | 12 | a form built on more than one mesh where UFL expects one (a solver bug to fix) |
| other | 10 | 15 |  |
| unresolved name | 6 | 42 | an expression names something the compiler cannot resolve |
| mixed boolean/numeric predicate | 5 | 17 | a geometry predicate that mixes booleans and numbers (e.g. '(x > 0) * 2') |
| CSG geometry | 4 | 21 | constructive solid geometry subvolumes (pyvcell reads them as expression-less analytic ones) |
| crash | 4 | 15 | a crash (not a clean refusal) |
| realization | 3 | 8 | geometry realization (meshing) failed |
| not implemented (other) | 1 | 1 | another feature not implemented yet |

## VCell's gate vs the run

VCell's pre-run check would offer FEniCSx for 578 of 600 applications; **385 of those then fail in the CLI** (a user finds out from a failed run, not an up-front refusal). By category:

| category | offered but fails |
|---|---|
| FastSystem | 116 |
| timeout | 91 |
| VCell cannot generate the math | 66 |
| mesh too large | 39 |
| memory | 22 |
| math validation | 11 |
| multiple domains (UFL) | 11 |
| other | 10 |
| unresolved name | 6 |
| mixed boolean/numeric predicate | 5 |
| crash | 4 |
| realization | 3 |
| not implemented (other) | 1 |

## Individual messages (normalized, ranked)

| reason | applications | simulations | example |
|---|---|---|---|
| RealizationError: image geometry '…' at h = # would need ~# tetrahedra (the limit is #); use h ≥ # | 30 | 169 | 101232348 / experiment |
| no regenerated VCML: failed to generate math: generated an invalid mathDescription: Initial condition for variable '…' references variable '…'. Initial conditio | 12 | 240 | 39669796 / Spatial 1 - 15nm shell - PIP2 seq at PSD |
| no regenerated VCML: failed to generate math: exception updating sizes | 12 | 29 | 79151545 /  Spatial fried egg no GDI |
| FormalismValidationError: MathDescription failed validation with # error(s): | 11 | 99 | 169992386 / Monkeyflower_pigmentation_v2 |
| ValueError: Found multiple domains, cannot return just one. | 11 | 12 | 243848510 / trans10X3 |
| no regenerated VCML: failed to generate math: '…' is either not found in your model or is not allowed to be used in the current context. Check that you have pro | 10 | 123 | 11006563 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_129925992.vcml: application '…': the math has a FastSystem (VCell'…'subdomain1'; FEniCSx does not solve fast systems, and ig | 10 | 13 | 129925992 / Tspine_1stOct_head_BARg |
| RealizationError: analytic geometry '…' at h = # would need ~# tetrahedra (the limit is #); use h ≥ # | 9 | 28 | 22681429 / Biophysical Letters |
| no regenerated VCML: failed to generate math: species '…' interacts with surface '…', but is not mapped spatially adjacent | 9 | 51 | 35512127 / Spatial 1 - 20nm shell |
| vcml_biomodels/regenerated/biomodel_53073020.vcml: application '…': the math has a FastSystem (VCell'…'subdomain1'; FEniCSx does not solve fast systems, and ign | 9 | 10 | 53073020 / Trans_30sept_neck |
| no regenerated VCML: failed to generate math: structure cyt not mapped, or unsupported GeometryClass null | 8 | 91 | 125971612 / 2D pde analytic |
| NotImplementedError: realization of dim=# geometry '…' is not implemented yet (supports dim #, #, #) | 8 | 18 | 31628467 / FRAP |
| vcml_biomodels/regenerated/biomodel_96774449.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ign | 7 | 7 | 96774449 / 3D_5um_1Hz |
| vcml_biomodels/regenerated/biomodel_32568081.vcml: application '…': the math has a FastSystem (VCell'…'C'; FEniCSx does not solve fast systems, and ignoring one | 6 | 10 | 32568081 / Furaptra |
| CompileError: unresolved name '…' (not in the compile context) | 6 | 42 | 52337206 / Simple 2D, Prof10, Cof01, Cap01, Fully Implicit |
| vcml_biomodels/regenerated/biomodel_27706976.vcml: application '…': the math has a FastSystem (VCell'…'C'; FEniCSx does not solve fast systems, and ignoring one | 5 | 14 | 27706976 / Furaptra |
| vcml_biomodels/regenerated/biomodel_28651971.vcml: application '…': the math has a FastSystem (VCell'…'cyto'; FEniCSx does not solve fast systems, and ignoring  | 5 | 11 | 28651971 / FigS5-fret1 |
| RvachevLoweringError: cannot lower a predicate that mixes boolean and numeric operands under '…' | 4 | 16 | 101637306 / 3D             |
| vcml_biomodels/regenerated/biomodel_129925992.vcml: application '…': the math has a FastSystem (VCell'…'PixelClass2'; FEniCSx does not solve fast systems, and i | 4 | 5 | 129925992 / Fig4_neuro2 |
| Error: error code # | 4 | 15 | 174183882 / Copy of Application1 |
| vcml_biomodels/regenerated/biomodel_32568171.vcml: application '…': the math has a FastSystem (VCell'…'C'; FEniCSx does not solve fast systems, and ignoring one | 4 | 6 | 32568171 / SyGCaMP2_1AP |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   Size parameter is not set. Size parameter is not set. Size parameter  | 4 | 10 | 34779681 / spatial_image |
| vcml_biomodels/regenerated/biomodel_83685534.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ign | 4 | 6 | 83685534 / 3D_1um_2s |
| NotImplementedError: multi-compartment realization of a 1D geometry is not implemented | 3 | 9 | 10014084 / 1D |
| RealizationError: 3D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 3 | 17 | 108026026 / AbsorbableKon |
| NgException: meshing failed | 3 | 3 | 252674703 / Copy of Application0 |
| RealizationError: geometry '…': subvolume(s) ['…'] are thinner than the mesh size h = #; use a smaller h | 3 | 8 | 270453508 / Application0 |
| vcml_biomodels/regenerated/biomodel_129925992.vcml: application '…': the math has a FastSystem (VCell'…'cyto'; FEniCSx does not solve fast systems, and ignoring | 2 | 5 | 129925992 / NO PKA-PTP |
| vcml_biomodels/regenerated/biomodel_13737035.vcml: application '…': the math has a FastSystem (VCell'…'cytosol'; FEniCSx does not solve fast systems, and ignori | 2 | 4 | 13737035 / NE_Full_N1E_demo |
| vcml_biomodels/regenerated/biomodel_18541949.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 2 | 7 | 18541949 / volt&calcium spatial |
| SolveError: the source term is non-finite (NaN/Inf) at the initial condition (t = #), before any step — a division by a quantity that is zero at t=#, or a fract | 2 | 3 | 32158617 / Spatial |
| vcml_biomodels/regenerated/biomodel_32568356.vcml: application '…': the math has a FastSystem (VCell'…'C'; FEniCSx does not solve fast systems, and ignoring one | 2 | 3 | 32568356 / SyGCaMP2_1AP |
| no regenerated VCML: failed to generate math: Application  endosome1 : H_ion_endosome has spatially resolved flux at membrane endosome, but doesn't diffuse in c | 2 | 3 | 32885648 / endosome12 |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   VCell spatial stochastic models only support 3D geometry.   See the P | 2 | 3 | 50216219 / pde_2D_shape |
| vcml_biomodels/regenerated/biomodel_53073020.vcml: application '…': the math has a FastSystem (VCell'…'cyto'; FEniCSx does not solve fast systems, and ignoring  | 2 | 2 | 53073020 / dia6_pka_diff |
| vcml_biomodels/regenerated/biomodel_53073020.vcml: application '…': the math has a FastSystem (VCell'…'PixelClass2'; FEniCSx does not solve fast systems, and ig | 2 | 2 | 53073020 / NewGoe_ISO_Nov10_flux |
| vcml_biomodels/regenerated/biomodel_60227051.vcml: application '…': the math has a FastSystem (VCell'…'PixelClass2'…'PixelClass3'; FEniCSx does not solve fast s | 2 | 2 | 60227051 / clust pic double |
| no regenerated VCML: failed to generate math: structure nuc not mapped, or unsupported GeometryClass null | 2 | 2 | 6081851 / Ascaris |
| vcml_biomodels/regenerated/biomodel_79296437.vcml: application '…': the math has a FastSystem (VCell'…'Cell'; FEniCSx does not solve fast systems, and ignoring  | 2 | 4 | 79296437 / Image-based 2D; Fast IP3PH |
| vcml_biomodels/regenerated/biomodel_9413015.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 2 | 12 | 9413015 / spatial_analyt |
| no regenerated VCML: failed to generate math: Application  Spatial : r_cytosol has spatially resolved flux at membrane cytosol, but doesn't diffuse in compartme | 1 | 1 | 11708030 / Spatial |
| RealizationError: 2D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 1 | 4 | 117544101 / Application0 |
| vcml_biomodels/regenerated/biomodel_148700996.vcml: application '…': the math has a FastSystem (VCell'…'Cytosol'…'Nucleus'; FEniCSx does not solve fast systems, | 1 | 3 | 148700996 / 2DGeom uniform ER |
| vcml_biomodels/regenerated/biomodel_159058523.vcml: application '…': the math has a FastSystem (VCell'…'Cytosol'…'Nucleus'; FEniCSx does not solve fast systems, | 1 | 3 | 159058523 / 2DGeom uniform ER |
| vcml_biomodels/regenerated/biomodel_17283953.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 1 | 6 | 17283953 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_17318643.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 1 | 6 | 17318643 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_18151359.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 1 | 1 | 18151359 / volt&calcium spatial clamped |
| no regenerated VCML: failed to generate math: required System property '…' not defined | 1 | 7 | 188175029 / Random Noise |
| vcml_biomodels/regenerated/biomodel_193968929.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ig | 1 | 5 | 193968929 / NWASP at Lam Tip in 3D Geometry |
| vcml_biomodels/regenerated/biomodel_194148251.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ig | 1 | 5 | 194148251 / NWASP at Lam Tip in 3D Geometry |
| vcml_biomodels/regenerated/biomodel_194193152.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ig | 1 | 6 | 194193152 / NWASP at Lam Tip in 3D Geometry |
| vcml_biomodels/regenerated/biomodel_209284198.vcml: application '…': the math has a FastSystem (VCell'…'region1'…'region2'; FEniCSx does not solve fast systems, | 1 | 2 | 209284198 / Microinjection - 3D |
| vcml_biomodels/regenerated/biomodel_21289199.vcml: application '…': the math has a FastSystem (VCell'…'cell'; FEniCSx does not solve fast systems, and ignoring  | 1 | 1 | 21289199 / double_size_B |
| vcml_biomodels/regenerated/biomodel_21462491.vcml: application '…': the math has a FastSystem (VCell'…'Bouton'; FEniCSx does not solve fast systems, and ignorin | 1 | 2 | 21462491 / DoubleSizeBouton |
| no regenerated VCML: failed to generate math: org.vcell.util.ConfigurationException: required System property '…' not defined | 1 | 1 | 222669359 / 3D |
| vcml_biomodels/regenerated/biomodel_22403233.vcml: application '…': the math has a FastSystem (VCell'…'subVolume2'…'subVolume0'; FEniCSx does not solve fast sys | 1 | 1 | 22403233 / Spatial |
| vcml_biomodels/regenerated/biomodel_22403244.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'…'subVolume2'; FEniCSx does not solve fast sys | 1 | 2 | 22403244 / spatial |
| vcml_biomodels/regenerated/biomodel_22403250.vcml: application '…': the math has a FastSystem (VCell'…'subVolume2'…'subVolume0'; FEniCSx does not solve fast sys | 1 | 1 | 22403250 / Spatial |
| vcml_biomodels/regenerated/biomodel_22463618.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ign | 1 | 2 | 22463618 / Localize NWASP to cell body |
| vcml_biomodels/regenerated/biomodel_23536676.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'…'subVolume2'; FEniCSx does not solve fast sys | 1 | 4 | 23536676 / wholecell |
| vcml_biomodels/regenerated/biomodel_23556394.vcml: application '…': the math has a FastSystem (VCell'…'cell'; FEniCSx does not solve fast systems, and ignoring  | 1 | 10 | 23556394 / sphere |
| vcml_biomodels/regenerated/biomodel_23624730.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'…'subVolume0'; FEniCSx does not solve fast sys | 1 | 12 | 23624730 / wholecell |
| vcml_biomodels/regenerated/biomodel_23624810.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'…'subVolume0'; FEniCSx does not solve fast sys | 1 | 15 | 23624810 / wholecell |
| vcml_biomodels/regenerated/biomodel_23725313.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 1 | 1 | 23725313 / spatial_analyt |
| NotImplementedError: region variable '…' lives on '…', realized as # disconnected regions; one value per region (per-region instancing) is not supported yet (## | 1 | 1 | 27010848 / Binding_Reaction |
| vcml_biomodels/regenerated/biomodel_28625428.vcml: application '…': the math has a FastSystem (VCell'…'cyto'; FEniCSx does not solve fast systems, and ignoring  | 1 | 3 | 28625428 / NO PKA-PTP |
| vcml_biomodels/regenerated/biomodel_28625786.vcml: application '…': the math has a FastSystem (VCell'…'cyto'; FEniCSx does not solve fast systems, and ignoring  | 1 | 7 | 28625786 / simple_3 |
| vcml_biomodels/regenerated/biomodel_315107768.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ig | 1 | 4 | 315107768 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_31564779.vcml: application '…': the math has a FastSystem (VCell'…'Cyto'; FEniCSx does not solve fast systems, and ignoring  | 1 | 1 | 31564779 / twoTransNoGap |
| SolveError: the residual became non-finite (NaN/Inf) at t ≈ # — likely a blow-up (e.g. an autocatalytic source growing without bound), a division by a quantity  | 1 | 3 | 31910148 / pde_dumble shape |
| vcml_biomodels/regenerated/biomodel_32158685.vcml: application '…': the math has a FastSystem (VCell'…'cytosol'; FEniCSx does not solve fast systems, and ignori | 1 | 1 | 32158685 / 3D |
| RvachevLoweringError: '…' is not allowed in a Rvachev implicit function (measure-zero set) | 1 | 1 | 33550815 / simulation_frap_echange |
| vcml_biomodels/regenerated/biomodel_3799924.vcml: application '…': the math has a FastSystem (VCell'…'cell'; FEniCSx does not solve fast systems, and ignoring o | 1 | 3 | 3799924 / sphere |
| vcml_biomodels/regenerated/biomodel_38086434.vcml: application '…': the math has a FastSystem (VCell'…'subVolume1'; FEniCSx does not solve fast systems, and ign | 1 | 4 | 38086434 / CALI, VASP=0, fine mesh |
| vcml_biomodels/regenerated/biomodel_42425810.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and ign | 1 | 5 | 42425810 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_54101111.vcml: application '…': the math has a FastSystem (VCell'…'cell'; FEniCSx does not solve fast systems, and ignoring  | 1 | 5 | 54101111 / spatial |
| no regenerated VCML: failed to generate math: Unable to perform operation. Errors found:   Clamped Species must be continuous rather than particles. Clamped Spe | 1 | 4 | 55073148 / spatial |
| RecursionError: maximum recursion depth exceeded | 1 | 3 | 5771185 / runner |
| no regenerated VCML: failed to generate math: structure out not mapped, or unsupported GeometryClass null | 1 | 1 | 6085954 / C_elegans |
| IndexError: list index out of range | 1 | 1 | 61710658 / Application0 |
| TypeError: unsupported operand type(s) for +: '…' and '…' | 1 | 1 | 65752152 / All_ACies_Active1 |
| vcml_biomodels/regenerated/biomodel_7537810.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 5 | 7537810 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_7752447.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 5 | 7752447 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_7752867.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 6 | 7752867 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_7753392.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 6 | 7753392 / volt&calcium spatial |
| no regenerated VCML: both volume and membrane functions must be defined for ConvolutionDataGenerator | 1 | 3 | 78352114 / Spatial fried egg |
| vcml_biomodels/regenerated/biomodel_79294095.vcml: application '…': the math has a FastSystem (VCell'…'Cell'; FEniCSx does not solve fast systems, and ignoring  | 1 | 2 | 79294095 / testbleach |
| ValueError: Dimension mismatch in dot product. | 1 | 1 | 84085279 / Determistic3D |
| vcml_biomodels/regenerated/biomodel_9143556.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 6 | 9143556 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_9297277.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 6 | 9297277 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_9297277.vcml: application '…': the math has a FastSystem (VCell'…'cytosol'; FEniCSx does not solve fast systems, and ignorin | 1 | 1 | 9297277 / app3d |
| vcml_biomodels/regenerated/biomodel_9368292.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 5 | 9368292 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_9368488.vcml: application '…': the math has a FastSystem (VCell'…'Nucleus'…'Cytosol'; FEniCSx does not solve fast systems, a | 1 | 1 | 9368488 / spatial |
| vcml_biomodels/regenerated/biomodel_9502781.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 7 | 9502781 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_9529236.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 5 | 9529236 / spatial_analyt |
| vcml_biomodels/regenerated/biomodel_9529712.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 7 | 9529712 / volt&calcium spatial |
| vcml_biomodels/regenerated/biomodel_9668730.vcml: application '…': the math has a FastSystem (VCell'…'cell'; FEniCSx does not solve fast systems, and ignoring o | 1 | 9 | 9668730 / sphere |
| vcml_biomodels/regenerated/biomodel_9734790.vcml: application '…': the math has a FastSystem (VCell'…'subVolume0'; FEniCSx does not solve fast systems, and igno | 1 | 5 | 9734790 / spatial_analyt |

