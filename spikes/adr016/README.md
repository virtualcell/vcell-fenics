# ADR 016 spike: whole-partition remeshing (recovered 2026-11-11)

This is the spike behind ADR 016 (vcell-fenics #231). It was uncommitted in the `vcell-fenics-remesh`
worktree when the laptop crashed (about 2026-11-10 18:18Z). The driver lived in a session scratchpad
that was lost; `remesh_spike.py` is rebuilt from the session transcript, with the session's two later
edits applied: `remesh_on_tangle=True` with per-output progress lines, and the stop after a fresh mesh
tangles on its first move. **This is spike code, not for merge.**

## What is here

- `src/vcell_fenics/backend/partition_remesh.py`: the remesh from the moved partition's boundary
  skeleton (Netgen), the bulk transfer (supermesh L2 projection, with a lumped non-negative fallback)
  and the exact membrane transfer.
- `backend/multi_compartment.py`: the hooks the remesh needs: start from `t_start` with `initial`
  fields, an entry point for a `pending_move` (second half-move), undo and early return on a tangle
  (`remesh_on_tangle`, result fields `needs_remesh` and `pending_move`).
- `backend/realize.py`: the realizer fix (ADR 016 decision 7), which drops any traced vertex closer
  than h/4 to the previous one.
- `pyvcell_bridge/simtask.py`: **a hack**, the image-geometry refusal lifted (`if images and False`)
  so the image moving_bleb could run. Revert it before building anything from this.

## Inputs (lost; how to recreate)

The inputs were simtask XMLs in the lost scratchpad under `bleb/`:
- `SimID_294599466`: moving_bleb, image geometry (`vcml_biomodels/biomodel_294599465.vcml`), written
  by a temporary VCell JUnit test (`TmpBlebSimTask`), with `EndTime` set to 3.5.
- `SimID_900108`: the 8-tooth analytic recast, `SimID_900135`: the 35-tooth recast. Both had
  `EndTime` 3.5 (from the 2.0 versions 900008 and 900035), with the boundary
  r(θ)² = 2.5² + (1 + 0.4 sin nθ)² centred at (5, 5).

Run: `PYTHONPATH=src ../vcell-fenics/.pixi/envs/dev/bin/python spikes/adr016/remesh_spike.py SIMTASK [h]`

## Results (as recorded in ADR 016)

| Geometry | Without remeshing | With remeshing | What stops it |
|---|---|---|---|
| 8-tooth recast, h = 0.04 | tangled at t ≈ 2.0 | reaches t = 3.5, one remesh at t = 1.75 | nothing |
| Image geometry, h = 0.04 | inverted cells at t ≈ 1.5 | carried to t = 1.75, two remeshes | the front closes on itself (gap 0.008 < h/4) |
| 35-tooth recast, h = 0.1 | tangled at t ≈ 1.05 | t = 2.55 with the realizer fix, one remesh | the front folds; a fresh mesh tangles at once |

Conservation per remesh is mostly 1e-13 to 1e-10; the worst is 3.8e-7, from the transfer kernel's
absolute clipping cutoff, which must become relative to cell area. A remesh costs about 80 s at
133k cells, nearly all of it the pure-Python transfer.

## Plan at the crash (waiting for Jim to accept ADR 016)

Build order, if ADR 016 is accepted:
1. The contact stop, and the realizer fix as its own PR.
2. The skeleton remesh, the transfer with a relative clipping cutoff, and their tests.
3. The undo, the per-cell trigger (a quarter of a cell's as-built area), the stop for a front that
   cannot advance, and the segmented solver in the runner.

Scope is 2D and serial. 3D needs a 3D supermesh transfer (ADR 016, "Not decided here").
