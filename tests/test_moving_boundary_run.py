"""VCell moving-boundary simulations through the solver path (tracker "Moving boundaries", M3).

The real moving-boundary SimulationTask ``SimID_274641196``: a disk ``cell`` (radius 3, centre (5, 5))
inside an empty ``ec``, three diffusing species in the cell, and a front velocity
``(sin t, cos t)`` read from the membrane's ``<Velocity>``. The mesh translates rigidly with the front;
the species have no velocity of their own, so — VCell's semantics — they stay in the lab frame and the
front sweeps them: in the cell's frame they drift backwards and pile against the trailing membrane,
while the zero-total-flux front keeps every species' mass exactly.
Variants of the same document (small text edits) exercise a constant front component, remeshing
under a strongly deforming front, and a species-dependent front.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vcell_fenics.cli import main
from vcell_fenics.results import Bundle

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "simtask"
_MB = _FIXTURES / "SimID_274641196_0__0.simtask.xml"
# a real VCell moving-boundary BioModel (from the FRAP tutorial): a cleavage furrow pinches a 2D cell at
# y = 0 (front velocity −exp(−y²/0.25)·tanh(x/5) in x); its species' velocity is routed through functions
# to a dotted constant (vobj_Cyt1_velX → vproc_1.velocityX = 0), as VCell writes every moving-boundary model
_FURROW = _FIXTURES / "furrow_SimID_1486629996_0__0.simtask.xml"
# the same furrow in 3D (hand-built from the 2D task): a sphere x² + y² + z² < 30 pinched by an axisymmetric
# contractile ring about the y axis, v = −exp(−y²/0.25)·tanh(ρ/5)·(x, 0, z)/ρ with ρ = √(x² + z²) — its xy
# cross-section is the 2D furrow's front
_FURROW_3D = _FIXTURES / "furrow3d_SimID_1486629996_0__0.simtask.xml"
_BUNDLE = "SimID_274641196_0_.fenics"
_SPECIES = ("C_cyt", "Ran_cyt", "RanC_cyt")
_VX = '<Function Name="sproc_0.velocityX" Domain="cell_ec_membrane">sin(t)</Function>'
_VY = '<Function Name="sproc_0.velocityY" Domain="cell_ec_membrane">cos(t)</Function>'


def _run(tmp_path: Path, *args: str, vx: str | None = None, vy: str | None = None) -> Bundle:
    text = _MB.read_text()
    if vx is not None:
        text = text.replace(_VX, _VX.replace(">sin(t)<", f">{vx}<"))
    if vy is not None:
        text = text.replace(_VY, _VY.replace(">cos(t)<", f">{vy}<"))
    task = tmp_path / _MB.name
    task.write_text(text)
    assert main(["--simtask", str(task), "--out", str(tmp_path), *args]) == 0
    return Bundle.open(tmp_path / _BUNDLE)


def _shift(bundle: Bundle) -> np.ndarray:
    first, last = bundle.coords("cell", 0), bundle.coords("cell", len(bundle.times) - 1)
    return np.asarray(last[:, :2].mean(axis=0) - first[:, :2].mean(axis=0))


def _assert_mass_conserved(bundle: Bundle) -> None:
    for species in _SPECIES:
        totals = bundle.stats("cell", species)[:, 1]
        assert np.allclose(totals, totals[0], rtol=1e-12), (species, totals)


def test_the_cell_translates_with_its_front_and_keeps_its_mass(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    bundle = _run(tmp_path, "--vc-print-status")
    manifest = bundle.manifest
    assert manifest.status == "completed"
    assert manifest.profile == "segmented"
    (segment,) = manifest.segments
    assert (segment.motion, segment.count) == ("ale", 11)
    assert manifest.solver.options["time_integration"] == "backward_euler"

    # a rigid translation: every point moves by the same vector …
    first, last = bundle.coords("cell", 0), bundle.coords("cell", 10)
    shift = _shift(bundle)
    assert np.abs((last[:, :2] - first[:, :2]) - shift).max() < 1e-12
    # … which is backward Euler's quadrature of the front velocity, evaluated at each step's end time
    # (a time-frozen velocity would give (0, 1)); it converges to (1 − cos 1, sin 1) as dt → 0
    dt = 0.1
    expected = dt * np.array([sum(math.sin(dt * k) for k in range(1, 11)), sum(math.cos(dt * k) for k in range(1, 11))])
    assert np.allclose(shift, expected, atol=1e-12)
    _assert_mass_conserved(bundle)

    # stdout carries status markers only (VCell's scanner), one data marker per row; the CLI writes them
    # to a duplicate of file descriptor 1, so capture at that level
    out = capfd.readouterr().out.split()
    assert out
    assert all(line.startswith("[[[") and line.endswith("]]]") for line in out)
    assert sum(line.startswith("[[[data:") for line in out) == 11

    # swept, not carried: Ran_cyt (initially = y) piles against the trailing membrane — the front moves
    # up and to the right, so by t = 1 the field decreases along the motion (a carried field would still
    # rise with y, flattened only by diffusion)
    coords, ran = bundle.coords("cell", 10), bundle.field("cell", "Ran_cyt", 10)
    motion = np.array([math.sin(1.0), math.cos(1.0)])
    gradient = np.linalg.lstsq(
        np.column_stack([coords[:, :2] - coords[:, :2].mean(axis=0), np.ones(len(ran))]), ran, rcond=None
    )[0][:2]
    assert gradient @ motion < 0.0, gradient

    summary = json.loads((tmp_path / _BUNDLE / "provenance" / "summary.json").read_text())
    assert summary["run"]["backend"] == "ale"
    assert summary["source"]["moving_boundary"]["velocity_dependence"] == "prescribed"


@pytest.mark.integration
def test_the_translation_converges_to_the_exact_displacement(tmp_path: Path) -> None:
    shift = _shift(_run(tmp_path, "--dt", "0.005"))
    assert np.allclose(shift, [1.0 - math.cos(1.0), math.sin(1.0)], atol=5e-3)


def test_a_constant_front_component(tmp_path: Path) -> None:
    # regression: a constant component (velocityY = 0) reached through two functions used to name a
    # parameter the imported model never defined
    bundle = _run(tmp_path, vx="0.5", vy="0.0")
    assert np.allclose(_shift(bundle), [0.5, 0.0], atol=1e-12)
    _assert_mass_conserved(bundle)


def test_a_deforming_front_remeshes_and_keeps_its_mass(tmp_path: Path) -> None:
    # v = (0.5 (x − 5)², 0) stretches the cell's right side and squeezes its left: the moving mesh degrades
    # and is replaced (a new bundle segment per remesh), the species carried over conservatively
    bundle = _run(tmp_path, vx="(0.5 * ((x - 5.0) ^ 2))", vy="0.0")
    segments = bundle.manifest.segments
    assert len(segments) >= 2
    assert [s.prefix for s in segments[1:]] == [f"seg{k:04d}/" for k in range(1, len(segments))]
    assert sum(s.count for s in segments) == len(bundle.times) == 11
    _assert_mass_conserved(bundle)
    for row in range(len(bundle.times)):
        assert bundle.coords("cell", row).shape[0] == bundle.field("cell", "C_cyt", row).shape[0]


def test_a_species_dependent_front(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # the front's speed follows a species (evaluated one step behind): the run completes and conserves
    bundle = _run(tmp_path, vx="(0.01 * C_cyt)", vy="0.0")
    assert "previous step" in capsys.readouterr().err
    assert _shift(bundle)[0] > 0.0  # C_cyt = x > 0 everywhere in the cell, so the front moves right
    _assert_mass_conserved(bundle)


def test_the_cleavage_furrow_pinches_the_cell_and_keeps_its_mass(tmp_path: Path) -> None:
    # coarse and short, but past the first remesh (t = 2 at h = 0.3), so the conservative remap runs: the
    # full model (h ≈ 0.1, 30 s) takes ~10 minutes and remeshes 8 times
    assert main(["--simtask", str(_FURROW), "--out", str(tmp_path), "--h", "0.3", "--t-final", "3.0"]) == 0
    bundle = Bundle.open(tmp_path / "SimID_1486629996_0_.fenics")
    assert bundle.manifest.status == "completed"
    segments = bundle.manifest.segments
    assert len(segments) >= 2 and all(s.motion == "ale" for s in segments)
    stats = bundle.stats("Cyt", "Dex")
    totals, area = stats[:, 1], stats[:, 1] / stats[:, 0]
    assert np.allclose(totals, totals[0], rtol=1e-12)
    assert area[-1] < area[0]  # the furrow ingresses


def test_the_3d_furrow_pinches_the_sphere_and_keeps_its_mass(tmp_path: Path) -> None:
    # coarse and short (no remesh yet): the ring moves the equator inward and the swept species is conserved
    assert main(["--simtask", str(_FURROW_3D), "--out", str(tmp_path), "--h", "1.0", "--t-final", "1.0"]) == 0
    bundle = Bundle.open(tmp_path / "SimID_1486629996_0_.fenics")
    assert bundle.manifest.status == "completed"
    (segment,) = bundle.manifest.segments
    assert segment.motion == "ale"
    first, last = bundle.coords("Cyt", 0), bundle.coords("Cyt", len(bundle.times) - 1)
    assert first.shape[1] == 3 and first.shape == last.shape
    stats = bundle.stats("Cyt", "Dex")
    totals, volume = stats[:, 1], stats[:, 1] / stats[:, 0]
    assert np.allclose(totals, totals[0], rtol=1e-12)
    assert volume[-1] < volume[0]
    # the equator (|y| small) moves toward the axis; the poles (|y| large) do not
    waist = np.abs(first[:, 1]) < 0.3
    radius = np.hypot(first[:, 0], first[:, 2]), np.hypot(last[:, 0], last[:, 2])
    assert radius[1][waist].max() < radius[0][waist].max() - 0.3
    poles = np.abs(first[:, 1]) > 4.5
    assert np.allclose(last[poles], first[poles], atol=1e-6)


def test_the_3d_furrow_remeshes_and_keeps_its_mass(tmp_path: Path) -> None:
    # past the first 3D remesh (t ≈ 1 at h = 1): new tetrahedral segments, the mass carried over exactly
    argv = [
        "--simtask",
        str(_FURROW_3D),
        "--out",
        str(tmp_path),
        "--h",
        "1.0",
        "--t-final",
        "3.0",
        "--output-dt",
        "0.5",
    ]
    assert main(argv) == 0
    bundle = Bundle.open(tmp_path / "SimID_1486629996_0_.fenics")
    segments = bundle.manifest.segments
    assert len(segments) >= 2 and all(s.motion == "ale" for s in segments)
    stats = bundle.stats("Cyt", "Dex")
    totals, volume = stats[:, 1], stats[:, 1] / stats[:, 0]
    assert np.allclose(totals, totals[0], rtol=1e-12)
    assert np.all(np.diff(volume) < 0.0)  # the ring keeps squeezing; remeshing does not jump the volume
    for row in range(len(bundle.times)):
        assert bundle.coords("Cyt", row).shape[0] == bundle.field("Cyt", "Dex", row).shape[0]
