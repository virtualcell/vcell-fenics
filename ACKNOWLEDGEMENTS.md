# Acknowledgements

vcell-fenics implements published methods, follows published models, and builds on open-source
software. This file names that work and says how each item is used, under the citation rule in
`CLAUDE.md`.

Each entry is marked:
- **Read**: the work itself was read in a session (the PDF, or its full text).
- **Not read here**: cited from standard knowledge or another paper.

Bibliographic details were checked against Crossref or DataCite (2026-11-13), except where marked
otherwise. The list was compiled from the repository and from the session transcripts since May 2026,
on 2026-11-13. Corrections are welcome.

## The Virtual Cell

vcell-fenics is a solver for the Virtual Cell, and much of it follows VCell's own semantics:
- geometry (`formalism/rvachev.py` ports VCell's R-function geometry code);
- the finite-volume semantics;
- the moving-boundary semantics.

VCell's solvers are its comparison baselines, and VCell users' models are its survey corpus.
- Schaff J, Fink CC, Slepchenko B, Carson JH, Loew LM. A general computational framework for
  modeling cellular structure and function. *Biophys. J.* 73:1135–1146 (1997).
  doi:10.1016/S0006-3495(97)78146-3.
- Schaff JC, Slepchenko BM, Choi Y-S, Wagner J, Resasco D, Loew LM. Analysis of nonlinear dynamics on
  arbitrary geometries with the Virtual Cell. *Chaos* 11:115–131 (2001). doi:10.1063/1.1350404.
- Loew LM, Schaff JC. The Virtual Cell: a software environment for computational cell biology.
  *Trends Biotechnol.* 19:401–406 (2001). doi:10.1016/s0167-7799(01)01740-1.
- Moraru II, Schaff JC, Slepchenko BM, et al. Virtual Cell modelling and simulation software
  environment. *IET Syst. Biol.* 2:352–362 (2008). doi:10.1049/iet-syb:20080102.
- Cowan AE, Moraru II, Schaff JC, Slepchenko BM, Loew LM. Spatial modeling of cell signaling
  networks. *Methods Cell Biol.* 110:195–221 (2012). doi:10.1016/B978-0-12-388403-9.00008-4
  (from vcell-core's verified bibliography).

## Methods and models adopted

**Moving boundaries and remeshing**
- **The moving-boundary baseline's algorithm and semantics.** Novak IL, Slepchenko BM. A conservative
  algorithm for parabolic problems in domains with moving boundaries. *J. Comput. Phys.*
  270:203–213 (2014). doi:10.1016/j.jcp.2014.03.014.
  - *Read* (2026-06-24).
  - *Use:* the mbsolver comparison baseline; lab-frame species, the front condition and exact
    conservation (ADR 013, 014; `cross_validation/`).
- **Supermesh conservative transfer** (`core/supermesh.py`, `core/bulk_remap.py`, the surface remap,
  ADR 016).
  - Farrell PE, Piggott MD, Pain CC, Gorman GJ, Wilson CR. Conservative interpolation between
    unstructured meshes via supermesh construction. *Comput. Methods Appl. Mech. Eng.*
    198:2632–2642 (2009). doi:10.1016/j.cma.2009.03.004.
  - Farrell PE, Maddison JR. Conservative interpolation between volume meshes by local Galerkin
    projection. *Comput. Methods Appl. Mech. Eng.* 200:89–100 (2011). doi:10.1016/j.cma.2010.07.015.
  - *Not read here.* libsupermesh's source was read (ADR 016).
- **ALE and the geometric conservation law** (ADR 015). *Abstracts only* (2026-11-10).
  - Hirt CW, Amsden AA, Cook JL. An arbitrary Lagrangian–Eulerian computing method for all flow
    speeds. *J. Comput. Phys.* 14:227–253 (1974). doi:10.1016/0021-9991(74)90051-5.
  - Farhat C, Geuzaine P, Grandmont C. The discrete geometric conservation law and the nonlinear
    stability of ALE schemes for the solution of flow problems on moving grids. *J. Comput. Phys.*
    174:669–694 (2001). doi:10.1006/jcph.2001.6932.
  - Formaggia L, Nobile F. A stability analysis for the arbitrary Lagrangian Eulerian formulation
    with finite elements. *East-West J. Numer. Math.* 7 (1999). Not in Crossref; the author order is
    to be checked against the published record.
- **The symmetric half-move split** (ADR 015). Strang G. On the construction and comparison of
  difference schemes. *SIAM J. Numer. Anal.* 5:506–517 (1968). doi:10.1137/0705041. *Not read here.*

**Flow and phase field**
- **Weak slip conditions** (`backend/slip.py`, `stokes.py`, `multiphase.py`). Nitsche J. Über ein
  Variationsprinzip zur Lösung von Dirichlet-Problemen bei Verwendung von Teilräumen, die keinen
  Randbedingungen unterworfen sind. *Abh. Math. Sem. Univ. Hamburg* 36:9–15 (1971).
  doi:10.1007/bf02995904. *Not read here.*
- **P2/P1 Stokes elements** (`backend/stokes.py`, `fsi.py`, `multiphase.py`). Taylor C, Hood P. A
  numerical solution of the Navier–Stokes equations using the finite element technique. *Comput.
  Fluids* 1:73–100 (1973). doi:10.1016/0045-7930(73)90027-3. *Not read here.*
- **H(div)-conforming Stokes** (`backend/stokes_hdiv.py`). Brezzi F, Douglas J, Marini LD. Two families
  of mixed finite elements for second order elliptic problems. *Numer. Math.* 47:217–235 (1985).
  doi:10.1007/bf01389710. *Not read here.*
- **Cahn–Hilliard phase field and its convex splitting** (`backend/cahn_hilliard.py`). *Not read here.*
  - Cahn JW, Hilliard JE. Free energy of a nonuniform system. I. Interfacial free energy. *J. Chem.
    Phys.* 28:258–267 (1958). doi:10.1063/1.1744102.
  - Eyre DJ. Unconditionally gradient stable time marching the Cahn–Hilliard equation. *MRS Proc.*
    529 (1998). doi:10.1557/proc-529-39.

**Geometry**
- **R-functions** (`formalism/rvachev.py`, a port of VCell's code). Shapiro V. Semi-analytic geometry with
  R-functions. *Acta Numerica* 16:239–303 (2007). doi:10.1017/s096249290631001x. A survey of Rvachev's
  work. *Not read here.*
- **Image geometry** (ADR 012; `backend/label_surfaces.py`, `realize.py`). *Not read here.*
  - SurfaceNets: Gibson SFF. Constrained elastic surface nets: generating smooth surfaces from binary
    segmented data. MICCAI 1998, *LNCS* 1496:888–898. doi:10.1007/bfb0056277.
  - Smoothing: Taubin G. A signal processing approach to fair surface design. SIGGRAPH '95, 351–358.
    doi:10.1145/218380.218473.
- **Analytic-geometry contours** (`backend/realize.py`, through scikit-image). *Not read here.*
  - Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface construction algorithm.
    SIGGRAPH '87, 163–169. doi:10.1145/37401.37422.
  - Lewiner T, Lopes H, Vieira AW, Tavares G. Efficient implementation of marching cubes' cases with
    topological guarantees. *J. Graphics Tools* 8:1–15 (2003). doi:10.1080/10867651.2003.10487582.
  - Douglas DH, Peucker TK. Algorithms for the reduction of the number of points required to
    represent a digitized line or its caricature. *Cartographica* 10:112–122 (1973).
    doi:10.3138/fm57-6770-u75u-7727.
- **Polygon clipping** (`core/bulk_remap.py`). Sutherland IE, Hodgman GW. Reentrant polygon clipping.
  *Commun. ACM* 17:32–42 (1974). doi:10.1145/360767.360802. *Not read here.*

**Verification and solvers**
- **Manufactured solutions** (`mms/`). Roache PJ. Code verification by the method of manufactured
  solutions. *J. Fluids Eng.* 124:4–10 (2002). doi:10.1115/1.1436090. *Not read here.*
- **Matching VCell's linear solvers** (`backend/linear_solvers.py`). Hindmarsh AC, Brown PN, Grant KE,
  et al. SUNDIALS: suite of nonlinear and differential/algebraic equation solvers. *ACM Trans. Math.
  Softw.* 31:363–396 (2005). doi:10.1145/1089014.1089020. *Not read here.*

## Models and work that shaped the use cases

- **UC-A, the motile-cell use case.** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko
  BM, Mogilner A. A free-boundary model of a motile cell explains turning behavior. *PLoS Comput.
  Biol.* 13:e1005862 (2017). doi:10.1371/journal.pcbi.1005862. *Read* (2026-06-24).
- **UC-C, the endocytosis use case.**
  - Nickaeen M, Berro J, Pollard TD, Slepchenko BM. Actin assembly produces sufficient forces for
    endocytosis in yeast. *Mol. Biol. Cell* 30:2014–2024 (2019). doi:10.1091/mbc.E19-01-0059.
    *Read* (2026-06-24).
  - Nickaeen M, Berro J, Pollard TD, Slepchenko BM. A model of actin-driven endocytosis explains
    differences of endocytic motility in budding and fission yeast. *Mol. Biol. Cell* 33:ar16
    (2022). doi:10.1091/mbc.E21-07-0362. *Read* (first page only, 2026-06-26).
  - Berro J, Sirotkin V, Pollard TD. Mathematical modeling of endocytic actin patch kinetics in
    fission yeast. *Mol. Biol. Cell* 21:2905–2915 (2010). doi:10.1091/mbc.e10-06-0494.
    *Not read here.*
- **The planned active gel** (`docs/modeling/cell-mechanics/`). Kruse K, Joanny JF, Jülicher F, Prost J,
  Sekimoto K. Generic theory of active polar gels: a paradigm for cytoskeletal dynamics. *Eur. Phys.
  J. E* 16:5–16 (2005). doi:10.1140/epje/e2005-00002-5. *Not read here.*
- **The project's scientific North Star.** Contri, Massing, Rangamani. A finite element framework for
  bulk-surface coupled PDEs to solve moving boundary problems in biophysics. arXiv:2510.23459
  (2025). *Abstract only*; the full author list has not been checked.
- **Design references** (`docs/research/`): SMART (Rangamani lab); FESTIM v2.0 (arXiv:2509.24760).

## Software

- **The FEniCSx project.** DOLFINx, UFL and FFCx are LGPL-3.0-or-later; Basix is MIT (conda metadata,
  2026-11-13).
  - Baratta IA, Dean JP, Dokken JS, et al. DOLFINx: the next generation FEniCS problem solving
    environment. Zenodo (2023). doi:10.5281/zenodo.10447666.
  - Alnæs MS, Logg A, Ølgaard KB, Rognes ME, Wells GN. Unified form language. *ACM Trans. Math.
    Softw.* 40:1–37 (2014). doi:10.1145/2566630.
  - Scroggs MW, Dokken JS, Richardson CN, Wells GN. Construction of arbitrary order finite element
    degree-of-freedom maps on polygonal and polyhedral cell meshes. *ACM Trans. Math. Softw.*
    48:1–23 (2022). doi:10.1145/3524456.
- **PETSc and petsc4py** (BSD-2-Clause). Dalcin LD, Paz RR, Kler PA, Cosimo A. Parallel distributed
  computing using Python. *Adv. Water Resour.* 34:1124–1139 (2011).
  doi:10.1016/j.advwatres.2011.04.013.
- **MUMPS** (CeCILL-C), the direct solver. Amestoy PR, Duff IS, L'Excellent JY, Koster J. A fully
  asynchronous multifrontal solver using distributed dynamic scheduling. *SIAM J. Matrix Anal.
  Appl.* 23:15–41 (2001). doi:10.1137/s0895479899358194.
- **MPICH** and **mpi4py**, for MPI (ADR 003).
- **Netgen**, the production mesher (ADR 008). Schöberl J. NETGEN: an advancing front 2D/3D-mesh
  generator based on abstract rules. *Comput. Visual. Sci.* 1:41–52 (1997).
  doi:10.1007/s007910050004.
- **gmsh**, dev and test only (GPL; `LICENSING.md`, ADR 008). Geuzaine C, Remacle J-F. Gmsh: a 3-D
  finite element mesh generator with built-in pre- and post-processing facilities. *Int. J. Numer.
  Methods Eng.* 79:1309–1331 (2009). doi:10.1002/nme.2579.
- **libsupermesh** (LGPL-2.1), the conservative remesh transfer (ADR 016).
- **VTK** (BSD-3-Clause), SurfaceNets and the VTU writer.
- **scikit-image** (BSD-3-Clause). van der Walt S, Schönberger JL, Nunez-Iglesias J, et al.
  scikit-image: image processing in Python. *PeerJ* 2:e453 (2014). doi:10.7717/peerj.453.
- **NumPy** and **SciPy**.
  - Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. *Nature*
    585:357–362 (2020). doi:10.1038/s41586-020-2649-2.
  - Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific
    computing in Python. *Nat. Methods* 17:261–272 (2020). doi:10.1038/s41592-019-0686-2.
- **pyvista**, dev only. Sullivan C, Kaszynski A. PyVista: 3D plotting and mesh analysis through a
  streamlined interface for the Visualization Toolkit (VTK). *J. Open Source Softw.* 4:1450 (2019).
  doi:10.21105/joss.01450.
- **Other packages:** scifem (MIT), zarr (MIT), meshio, matplotlib, pyvcell and its pydantic/lxml/numexpr
  stack, Pixi and conda-forge, ruff, mypy, pytest.
- **Comparison baselines:** VCell's finite-volume solver and its FronTier-based moving-boundary solver,
  run through pyvcell in their own environment.

## Considered and not adopted

These are listed so that the design history is honest; none is used.
- Cut and trace FEM (CutFEMx, TraceFEM, multiphenicsx; Approach D is unbuilt).
- Phase-field multi-cell codes.
- IBAMR.
- The front-tracking codes named in `docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md`.
