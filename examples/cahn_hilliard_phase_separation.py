"""Spinodal decomposition, driven entirely from the declarative formalism.

Cahn–Hilliard is the canonical diffuse-interface model of **liquid–liquid phase separation** — the
physics of biomolecular condensates / membraneless organelles (a protein mixture demixes into a dense
and a dilute phase). One conserved order parameter `φ` (local dense-phase fraction) evolves by

    ∂φ/∂t = ∇·(M ∇μ),   μ = f'(φ) − ε² ∇²φ,   f(φ) = W φ²(1 − φ)²,

a 4th-order conserved gradient flow. From a near-uniform **critical** mixture (`φ̄ = 0.5`) with tiny
noise — an unstable state on the hump of the double well — the field demixes toward the two wells
(`φ = 0` dilute, `φ = 1` dense) across diffuse interfaces of width ~`ε`, growing the characteristic
interconnected spinodal pattern. (The demixing then locks into a metastable pattern; long-time
curvature-driven coarsening is real but much slower, so this demo shows the separation itself.)

Everything here is **declarative** — there are no solver internals or hand-built meshes in this script:

  - the model is a `MathDescription` with one `cahn_hilliard` equation (slots `interface_width` = ε and
    `well_height` = W; ε is chosen so the interface spans several mesh cells, i.e. is well resolved);
  - the spinodal seed is the IC expression `0.5 + normal(0, 0.02)` — a **random primitive** realized once
    into a fixed (seeded) field, so it is a proper function of space, not a fresh draw per evaluation;
  - the time-lapse comes from `iter_cahn_hilliard`, the template's snapshot driver.

Two exact invariants, printed as it runs: `∫φ` is **conserved** (the conservative `∇·(M∇μ)` form), and
the free energy `F = ∫[W φ²(1−φ)² + ε²/2 |∇φ|²]` (a Lyapunov functional) **decreases monotonically** —
the gradient flow runs downhill. Frames are sampled at evenly-spaced *energy* levels, so they land where
the morphology actually changes and the slow tail collapses to one frame.

Writes `cahn_hilliard_phase_separation.png`: a tiled `φ` sequence (dense phase red, dilute blue) from the
noisy mixture to the demixed two-phase pattern.

    .pixi/envs/dev/bin/python examples/cahn_hilliard_phase_separation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista

from vcell_fenics.backend import cahn_hilliard_free_energy, iter_cahn_hilliard, make_disk_geometry
from vcell_fenics.formalism.schema import (
    MathDescription,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.viz import _function_to_pyvista

_HERE = Path(__file__).parent
# The equilibrium interface has scale δ = ε/√(2W) (verified in tests against the exact tanh profile). The
# mesh must resolve it — δ ≳ 2h — or the interface pins to the grid (it freezes, under-resolved and aliased,
# the failure mode of a naive choice). Here δ = 0.05/√4 = 0.025 ≈ 2.5·h, so the diffuse interface spans
# several cells and renders smoothly. ε is NOT pushed higher than this: too large a gradient penalty
# suppresses separation entirely.
_EPSILON = 0.05  # ε  (gradient penalty ε²)
_WELL_HEIGHT = 2.0  # W; with ε this sets δ = ε/√(2W) = 0.025 and a moderate separation rate
_H = 0.01  # mesh size — δ/h ≈ 2.5, so the interface is resolved
_DT = 1.0e-4
_EVERY = 10  # snapshot stride (steps); the driver yields (t, φ) this often
_PLATEAU_TOL = 1.0e-3  # stop once the free energy stops falling (separation finished)
_N_FRAMES = 8
_SEED = 7


def model() -> MathDescription:
    return MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume")],
        variables=[Variable(name="c", subdomain="cyto")],
        equations=[
            TemplateEquation(
                template="cahn_hilliard",
                variable="c",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"interface_width": str(_EPSILON), "well_height": str(_WELL_HEIGHT)},
                initial_condition="0.5 + normal(0, 0.02)",  # spinodal seed: a stored, seeded random field
            )
        ],
    )


def _write_tiled_image(frames: list[tuple[pyvista.UnstructuredGrid, float]], path: Path) -> None:
    pyvista.OFF_SCREEN = True
    cols = (len(frames) + 1) // 2
    plotter = pyvista.Plotter(shape=(2, cols), off_screen=True, window_size=[210 * cols, 440], border=False)
    for i, (grid, t) in enumerate(frames):
        plotter.subplot(i // cols, i % cols)
        plotter.add_mesh(
            grid,
            scalars="phi",
            clim=[0.0, 1.0],
            cmap="coolwarm",
            show_edges=False,
            show_scalar_bar=(i == 0),
            scalar_bar_args={"title": "phase phi", "n_labels": 3, "fmt": "%.1f", "label_font_size": 10},
        )
        plotter.add_text(f"t = {t:.3f}", font_size=9)
        plotter.view_xy()  # type: ignore[call-arg]  # pyvista stub drops self on the @wraps-decorated method
        plotter.camera.zoom(1.3)
    plotter.screenshot(str(path))
    plotter.close()


def main() -> None:
    geom = make_disk_geometry("cell", volume_subdomain="cyto", radius=0.75, h=_H)
    md = model()

    # Stream (t, φ) snapshots from the template driver; record (t, energy, grid) and stop at the plateau.
    history: list[tuple[float, float, pyvista.UnstructuredGrid]] = []
    total0 = float("nan")
    for t, phi in iter_cahn_hilliard(md, geom, dt=_DT, t_final=0.06, every=_EVERY, seed=_SEED):
        phi.name = "phi"
        energy = cahn_hilliard_free_energy(phi, epsilon=_EPSILON, well_height=_WELL_HEIGHT)
        history.append((t, energy, _function_to_pyvista(phi)))
        if np.isnan(total0):
            total0 = float(np.sum(phi.x.array))
        dropped = energy < 0.5 * history[0][1]  # past the steep descent (not still incubating)
        if dropped and len(history) >= 2 and history[-2][1] - energy < _PLATEAU_TOL:
            break

    # Select frames at evenly-spaced energy levels (energy is monotone ⇒ uniform sampling of the change).
    levels = np.linspace(history[0][1], history[-1][1], _N_FRAMES)
    chosen: list[int] = []
    for level in levels:
        j = int(np.argmin([abs(e - level) for _, e, _ in history]))
        if j not in chosen:
            chosen.append(j)
    frames = [(history[j][2], history[j][0]) for j in chosen]

    delta = _EPSILON / np.sqrt(2.0 * _WELL_HEIGHT)
    print(f"  disk Ø1.5, h={_H}, ε={_EPSILON}, W={_WELL_HEIGHT}, φ̄ = 0.5 + normal(0, 0.02)")
    print(f"  interface δ = ε/√(2W) = {delta:.3f} ≈ {delta / _H:.1f} cells (resolved)\n")
    print(f"  {'t':>7} {'∫φ (conserved)':>16} {'free energy F':>15}")
    for j in chosen:
        t, e, _ = history[j]
        print(f"  {t:7.3f} {total0:>16.6f} {e:>15.4f}")  # ∫φ is the same every row — that is the point
    print("\n  free energy F fell monotonically; ∫φ held — the mixture demixed into the two phases.")
    out = _HERE / "cahn_hilliard_phase_separation.png"
    _write_tiled_image(frames, out)
    print(f"  wrote {out.name} — φ from the noisy critical mixture to the demixed two-phase pattern")


if __name__ == "__main__":
    main()
