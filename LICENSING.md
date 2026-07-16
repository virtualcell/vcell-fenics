# Licensing & third-party dependencies

`vcell-fenics` is distributed under the **MIT License** (see `LICENSE.md`). This document records the
license posture of its dependencies and, in particular, **why the package carries no GPL (copyleft)
obligation** — the practical goal being that `vcell-fenics` can be shipped and reused permissively.

> **Not legal advice.** This is an engineering record of how the project is structured to stay
> permissively licensable. The final dependency split and any distributed `NOTICE` wording should be
> reviewed by counsel before a formal release.

## Summary

| Concern | Status |
|---|---|
| This package's own code | MIT |
| Runtime dependency tree | Permissive (BSD/MIT) + **LGPL** only — no copyleft on our code |
| gmsh (GPL-2.0-or-later) | **Dev/test-only.** Not a runtime dependency, not imported by any `src/` module, never distributed |
| Production geometry mesher | **Netgen (LGPL-2.1)**, via `backend/realize.py` — not gmsh |

## The package (MIT)

`src/vcell_fenics/` is MIT-licensed. It ships a `py.typed` marker and is imported by downstream code
(e.g. the pyvcell integration). Nothing in `src/` imports a GPL library, so importing or linking
`vcell_fenics` imposes no copyleft obligation on the importer.

## Runtime dependencies

The runtime dependencies (conda-forge, declared under `[tool.pixi.dependencies]` in `pyproject.toml`)
are permissive or weak-copyleft (LGPL):

- **FEniCSx stack** — DOLFINx, UFL, FFCx, Basix: **LGPL-3.0**. Used by dynamic import / linkage.
- **Netgen / pyngcore**: **LGPL-2.1**. The body-fitted meshing engine behind `backend/realize.py`
  (the production geometry path) and the ALE remesher (`core/region_remesh_netgen.py`).
- **PETSc / petsc4py**: BSD-2-Clause. **NumPy, SciPy**: BSD-3-Clause. **pyvista**: MIT.
  **scikit-image**: BSD-3-Clause. **pydantic**: MIT. **meshio**: MIT. **matplotlib**: PSF/BSD-compatible.

**LGPL is compatible with shipping `vcell-fenics` under MIT.** The LGPL's copyleft attaches to
modifications *of the LGPL library itself* and to static linkage; using these libraries as installed,
dynamically imported dependencies — which is exactly how the DOLFINx/Netgen Python stack is consumed —
does not place any copyleft requirement on `vcell-fenics`'s own MIT code. We do not modify or vendor
these libraries; they are resolved from conda-forge at environment-creation time.

## gmsh isolation (GPL-2.0-or-later)

gmsh is **GPL-2.0-or-later**. To keep `vcell-fenics` permissively licensable, gmsh is quarantined:

1. **Not a runtime dependency.** gmsh and `python-gmsh` are declared **only** under
   `[tool.pixi.feature.dev.dependencies]`, never under `[tool.pixi.dependencies]`. The default
   (distributed) environment does not contain gmsh.
2. **Not imported by any `src/` module.** There is no `import gmsh` (nor `dolfinx.io.gmsh`) anywhere
   under `src/vcell_fenics/`. Importing `vcell_fenics`, `vcell_fenics.backend`, or `vcell_fenics.core`
   does not load gmsh. This is verifiable:

   ```bash
   grep -rn "import gmsh" src/vcell_fenics/            # → no matches
   python -c "import sys, vcell_fenics.backend; assert 'gmsh' not in sys.modules"
   ```

3. **The production geometry path uses Netgen, not gmsh.** `backend/realize.py` (the pyvcell-integration
   realization path) and the ALE remesher (`core/region_remesh_netgen.py`) are backed by LGPL Netgen.
4. **gmsh is used only in tests/dev.** The gmsh-based reference meshers live under
   `tests/gmsh_meshers/` (`create_disk`, `create_disk_with_membrane`, `create_cell_extracellular`,
   `create_extracellular_annulus`) and the gmsh region remesher (`tests/gmsh_meshers/region_remesh.py`).
   They are exercised by the test suite (which runs in the `dev` environment) as a cross-check against
   the Netgen production path, but are **never distributed and never required to run `vcell-fenics`.**

### Guardrails (do not regress)

- **Never** add gmsh (or `python-gmsh`) to `[tool.pixi.dependencies]`. It belongs in the `dev` feature.
- **Never** add `import gmsh` or `from dolfinx.io.gmsh import ...` to any module under `src/`. If a
  meshing capability is needed at runtime, use Netgen (`realize` / `region_remesh_netgen`).
- **Never** vendor gmsh source/binaries into this repository, and do not commit gmsh-generated mesh
  artifacts (`.msh` files) as distributed content.
- The gmsh reference meshers are retained under `tests/` purely so the meshing know-how is preserved and
  the Netgen path has a cross-check — treat them as test fixtures, not library code.

## Why this matters

The moving-boundary / meshing problem class often reaches for gmsh, and gmsh's OCC kernel is convenient.
But gmsh's GPL would force any distributed solver that *requires* it to be GPL too. By making Netgen the
runtime mesher and keeping gmsh strictly in the dev/test tier, `vcell-fenics` stays MIT-shippable while
retaining gmsh as a development convenience and correctness cross-check. See
`docs/decisions/008-gmsh-license-isolation.md` for the original decision and the Netgen contingency.
