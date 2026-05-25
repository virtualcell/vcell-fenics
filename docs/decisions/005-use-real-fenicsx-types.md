# ADR 005 — Use the FEniCSx stack's real types instead of `follow_imports = "skip"`

**Date:** 2026-05-24
**Status:** Accepted

## Context

The mypy config previously suppressed the entire FEniCSx scientific stack with `follow_imports = "skip"` (for `dolfinx.*`, `ufl.*`, `basix.*`, `petsc4py.*`, `pyvista.*`, etc.), on the belief that those packages shipped no usable type information. That makes mypy replace each such module with `Any` wholesale.

On inspection that premise was wrong: in the pinned environment **every one of those packages ships a `py.typed` marker** — `ufl`, `dolfinx`, `basix`, `ffcx`, `petsc4py`, `mpi4py` all advertise inline types (PEP 561). `skip` was therefore discarding real type information, not working around its absence. An experiment (removing `skip`) surfaced a **genuine latent bug it had been hiding**: `dolfinx.io.gmsh.model_to_mesh` returns `facet_tags: MeshTags | None`, and both disk-geometry builders dereferenced it (`data.facet_tags.find(...)`, `StaticDisk(facet_tags=data.facet_tags)`) without handling `None`.

The reason `skip` is tempting is real, though: the shipped types are *incomplete*. UFL marks itself `py.typed` but leaves most functions unannotated, so under `strict` mode's `disallow_untyped_calls` every `grad()`, `dot()`, `Measure()`, `TrialFunction()` call is flagged — about 30 such errors, pure noise. A handful of stubs are also *wrong* (petsc4py types `PETSc.ScalarType` as a non-callable dtype; pyvista's `cmap` is an enormous `Literal`, its `window_size` insists on `list`, and `view_xy` has a decorator stub bug).

## Decision

Stop skipping. Use the packages' real types, and suppress only the specific noise:

- Remove `follow_imports = "skip"`. Keep `ignore_missing_imports = true` for the listed modules so the genuinely untyped ones (`gmsh`, `meshio`) still resolve to `Any` without a "missing stubs" error.
- Add `untyped_calls_exclude = ["ufl", "basix", "dolfinx", "pyvista", "scifem", "dolfinx_mpc"]` to `[tool.mypy]`. This turns off **only** `disallow_untyped_calls` for those packages — so calling their unannotated functions is allowed — while every *other* strict check still runs against their real types.
- Where a stub is actually wrong, use a targeted `# type: ignore[code]` or `cast()` at that exact call site (petsc4py `ScalarType`; the three pyvista signatures), never a blanket `Any` in our own code.
- The `backend/_typing.py` aliases stay: they document which opaque object a field holds where the upstream type is still `Any`, and would repoint to real types if upstream completes its annotations.

## Consequences

**Positive:**

- Real type-checking against the FEniCSx API. The switch immediately caught the `MeshTags | None` dereference bug and a real `float | complex` scalar issue.
- The "skip hides everything" failure mode is gone; future genuine type errors at our FEniCSx call sites will surface.
- `disallow_untyped_calls` still protects calls to *our own* untyped code; it is relaxed only for the named third-party packages.

**Negative:**

- A class of false positives must be carried as targeted ignores (currently petsc4py `ScalarType`, three pyvista signatures). These need occasional review as upstream stubs change — a `warn_unused_ignores` failure will flag any that become unnecessary.
- Returns from unannotated FEniCSx functions are still `Any`, so checking is partial, not complete. `untyped_calls_exclude` is a blunt per-package switch, not per-function.
- mypy now analyzes the (large) third-party type stubs, marginally slower than skipping them.

## Notes

- Supersedes the typing guidance in the earlier `CLAUDE.md` "Quality enforcement" note, which is updated to match.
- This reverses a default I had set earlier in development on the mistaken "no usable stubs" premise; the `py.typed` evidence is what changed the decision.
- If a future DOLFINx/UFL release ships complete inline annotations, `untyped_calls_exclude` entries can be dropped package-by-package (and the `_typing.py` aliases repointed) for fuller checking.
