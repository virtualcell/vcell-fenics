# FEniCSx coverage of saved VCell biomodels

*A seeded random sample (seed 20260926) of 600 applications; counts below are over the sample, so percentages estimate the whole corpus.*

Corpus: 2510 biomodels, 31996 simulations, of which **15117 simulations in 3423 applications** are spatial, deterministic and saved with a VCell finite-volume-family solver. 600 of those applications were run through the FEniCSx CLI.

**Ran: 29 / 600 applications (5%), covering 94 / 2588 of their simulations.**

## By solver

| solver | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| Sundials Stiff PDE Solver (Variable Time Step) | 348 | 15 (4%) | 31 | 302 |
| Finite Volume, Regular Grid | 174 | 0 (0%) | 2 | 172 |
| Finite Volume Standalone, Regular Grid | 56 | 1 (2%) | 1 | 54 |
| Chombo Standalone | 17 | 8 (47%) | 0 | 9 |
| MovingB | 5 | 5 (100%) | 0 | 0 |

## By dimension

| dimension | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| 2D | 333 | 14 (4%) | 3 | 316 |
| 3D | 256 | 15 (6%) | 31 | 210 |
| 1D | 11 | 0 (0%) | 0 | 11 |

## By geometry

| geometry | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| analytic | 390 | 22 (6%) | 13 | 355 |
| image | 199 | 7 (4%) | 21 | 171 |
| mixed(analytic+csg) | 9 | 0 (0%) | 0 | 9 |
| csg | 2 | 0 (0%) | 0 | 2 |

## By moving boundary

| moving boundary | applications | ran | timeout / memory | failed |
|---|---|---|---|---|
| fixed | 596 | 25 (4%) | 34 | 537 |
| moving | 4 | 4 (100%) | 0 | 0 |

## Why applications don't run (by category, ranked)

| category | applications | simulations | what it means |
|---|---|---|---|
| legacy domain-less variable | 203 | 1024 | old-style VCell math: one volume variable on both sides of a membrane |
| one-sided membrane species | 79 | 303 | membrane species with bulk species in one compartment only (#183) |
| math validation | 59 | 204 | the imported math fails formalism validation |
| non-diffusing species (lumped_ode) | 48 | 265 | ODE (non-diffusing) species on a spatial domain |
| subvolume touches the box | 48 | 150 | a subvolume boundary touches the domain box (2D/3D realization) |
| 3+ compartments | 33 | 121 | species on three or more compartments, or compartments without a membrane between them |
| timeout | 31 | 162 | no result within the timeout at the simulation's own mesh |
| mesh too large | 22 | 154 | the simulation's own mesh size exceeds the FEM mesh limit (may run coarser) |
| other | 15 | 26 |  |
| not implemented (other) | 10 | 31 | another feature not implemented yet |
| unresolved name | 9 | 18 | an expression names something the compiler cannot resolve |
| 1D geometry | 8 | 18 | one-dimensional geometry |
| CSG geometry | 3 | 11 | constructive solid geometry subvolumes (pyvcell reads them as expression-less analytic ones) |
| memory | 3 | 7 | past the survey's per-run memory cap at the simulation's own mesh |

## VCell's gate vs the run

VCell's pre-run check would offer FEniCSx for 578 of 600 applications; **549 of those then fail in the CLI** (a user finds out from a failed run, not an up-front refusal). By category:

| category | offered but fails |
|---|---|
| legacy domain-less variable | 200 |
| one-sided membrane species | 77 |
| math validation | 59 |
| non-diffusing species (lumped_ode) | 48 |
| subvolume touches the box | 42 |
| 3+ compartments | 33 |
| timeout | 31 |
| mesh too large | 22 |
| other | 15 |
| not implemented (other) | 10 |
| unresolved name | 9 |
| memory | 3 |

## Individual messages (normalized, ranked)

| reason | applications | simulations | example |
|---|---|---|---|
| NotImplementedError: jump condition for '…' on membrane '…': this is a legacy domain-less volume variable defined on both sides of the membrane, so its inside / | 203 | 1024 | 10011510 / spatial |
| RunError: membrane species on '…' with bulk species only in '…': the membrane-coupled solver needs species in both compartments it separates ('…' has none) — no | 79 | 303 | 100786955 / 2D chombo |
| FormalismValidationError: MathDescription failed validation with # error(s): | 59 | 204 | 120428417 / combined_spatial |
| NotImplementedError: backend v1 supports templates ['…', '…'], not '…' | 48 | 265 | 10014084 / 2D distributed |
| NotImplementedError: 3D realization requires the boundary of '…' to be strictly inside the box (a subvolume touching the box boundary is a later slice) | 32 | 116 | 101963252 / extra: simple test geometry.  "few fingers" |
| RealizationError: image geometry '…' at h = # would need ~# tetrahedra (the limit is #); use h ≥ # | 22 | 154 | 101232348 / experiment |
| NotImplementedError: 2D realization requires the boundary of '…' to be strictly inside the box (a subvolume touching the box boundary is a later slice) | 16 | 34 | 109135045 / Application0 |
| KeyError: '…' | 10 | 20 | 109232682 / timed |
| CompileError: unresolved name '…' (not in the compile context) | 9 | 18 | 131370534 / 0.3x0.42-Diff-Ca-0.5-New6th-Spark1-Gain200-sigma0.0099-BC-XValue-YFlux |
| NotImplementedError: membrane coupling needs a bulk diffusion equation for '…' | 8 | 24 | 110218606 / NE_Full_new |
| NotImplementedError: realization of dim=# geometry '…' is not implemented yet (supports dim #, #, #) | 8 | 18 | 31628467 / FRAP |
| NgException: meshing failed | 3 | 3 | 252674703 / Copy of Application0 |
| RunError: the model'…'subVolume1'…'unnamed'…'Rectangle # x #'…'subVolume0_subVolume1_membrane']) | 3 | 27 | 62467093 / Actin Comet Tails |
| RealizationError: 3D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 2 | 7 | 108026026 / AbsorbableKon |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membrane']) | 2 | 6 | 78352114 / Spatial fried egg |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry5'…'subdomain0_subdomain1_membrane']) | 2 | 8 | 78709778 / 2D |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium861043460'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membrane | 2 | 2 | 78742497 /  Spatial fried egg pulse GEF |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium720792842'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membrane | 2 | 6 | 79240112 / Spatial fried egg |
| RunError: the model'…'Cell'…'unnamed'…'sphere for PIP1269194936'…'Cell_extracellular_membrane'…'Cell_Nucleus_membrane']) | 2 | 5 | 79294095 / testbleach |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium44531325'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membrane' | 2 | 6 | 79322430 /  Spatial fried egg no GDI |
| RunError: the model'…'subVolume1'…'unnamed'…'fried_egg_3'…'subVolume0_subVolume1_membrane'…'subVolume1_subVolume2_membrane']) | 2 | 6 | 81921927 / Biophysical Letters |
| NotImplementedError: interface coupling needs a bulk diffusion equation for each species on compartment '…' | 1 | 6 | 110197099 / spatial |
| RealizationError: 2D geometry '…' subvolume '…' must be '…' with an expression to be realized (got type '…') | 1 | 4 | 117544101 / Application0 |
| IndexError: list index out of range | 1 | 2 | 40636296 / 2D spatial=false |
| RunError: the model'…'cytosol'…'extracellular'…'GeoForBug14' declares no membrane between them (surfaces: none) | 1 | 4 | 4069662 / Spatial |
| RunError: the model'…'cell'…'unnamed'…'test2_lsm'…'cell_ec_membrane']) | 1 | 1 | 50216219 / pde_2D_shape |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry894097955'…'subdomain0_subdomain1_membrane']) | 1 | 3 | 52337206 / Simple 2D, Prof10, Cof01, Cap01, Fully Implicit |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry2059119022'…'subdomain0_subdomain1_membrane']) | 1 | 5 | 52337206 / Simple 3D, Prof10, Cof01, Cap01, Fully Implicit |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry346640342'…'subdomain0_subdomain1_membrane']) | 1 | 5 | 52341015 / Simple 3D, Prof10, Cof01, Cap01, Fully Implicit |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry1411620171'…'subdomain0_subdomain1_membrane']) | 1 | 4 | 52341015 / LP, Prof10, Cof0, Cap01, Fully Implicit |
| RunError: the model'…'subdomain0'…'unnamed'…'Geometry416557365'…'subdomain0_subdomain1_membrane']) | 1 | 2 | 52982138 / pde |
| RunError: the model'…'cyt'…'unnamed'…'3D image neuroblastoma585641587'…'cyt_ec_membrane'…'cyt_nuc_membrane']) | 1 | 2 | 55569686 / 3d image |
| RunError: the model'…'subVolume0'…'subVolume1'…'myGeo' declares no membrane between them (surfaces: none) | 1 | 3 | 5771185 / runner |
| RunError: the model'…'Cyt'…'unnamed'…'N1E_for_PIP1409613287'…'Cyt_EC_membrane'…'Cyt_Nuc_membrane']) | 1 | 2 | 72815317 / spatial2D |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium1645926951'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membran | 1 | 3 | 78709778 /  Spatial fried egg no GDI |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium1535731751'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membran | 1 | 3 | 79151545 /  Spatial fried egg no GDI |
| RunError: the model'…'subVolume1'…'unnamed'…'3D half cell with very thin lamelipodium1194601750'…'subVolume0_subVolume1_membrane'…'subVolume1_subdomain0_membran | 1 | 3 | 79178981 / Spatial fried egg |
| RunError: the model'…'Cell'…'unnamed'…'sphere for PIP477064015'…'Cell_extracellular_membrane'…'Cell_Nucleus_membrane']) | 1 | 10 | 79294095 / FLIP |
| RunError: the model'…'Cell'…'unnamed'…'sphere for PIP2'…'Cell_extracellular_membrane'…'Cell_Nucleus_membrane']) | 1 | 1 | 79296437 / Image-based 2D; Fast IP3PH |
| RunError: the model'…'subdomain1'…'unnamed'…'Geometry1547138152'…'subdomain0_subdomain1_membrane']) | 1 | 4 | 79317919 / 2D |
| RvachevLoweringError: cannot lower a predicate that mixes boolean and numeric operands under '…' | 1 | 1 | 95142058 / simple_transport_simulation_spatialGeom |
| NotImplementedError: Dirichlet BC on boundary '…': the membrane-coupled solver supports a reservoir Dirichlet only on the outer box boundary '…' | 1 | 1 | 97918549 / Application0 |

