"""A VCell model on an image geometry, end to end through the CLI (lowered math + geometry YAML in, a
results bundle out).

The geometry is VCell's segmented 3D tutorial image (256×256×34 pixels, ec ⊃ cytosol ⊃ Nucleus), voxels
carried in the YAML; the math is nucleocytoplasmic exchange — ``c`` in the cytosol, ``n`` in the Nucleus,
a permeability flux across the nuclear membrane (``cross_validation/image_nuclear_fv.py`` authors it and
compares the two solvers). The image is realized body-fitted and smoothed and the two compartments solved
by the interface-coupled method of lines, with ``ec`` dropped as the background."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vcell_fenics.cli import main
from vcell_fenics.results import Bundle

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "image"
_GEOMETRY = _FIXTURES / "tutorial_geom.yaml"


def test_an_image_geometry_model_runs_end_to_end(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    argv = ["--math", str(_FIXTURES / "nuclear_math.yaml"), "--geometry", str(_GEOMETRY)]
    assert main([*argv, "--t-final", "0.5", "--output-dt", "0.25", "--h", "2.0", "--out", str(tmp_path)]) == 0
    bundle = Bundle.open(tmp_path / "results.fenics")
    assert bundle.manifest.status == "completed"
    c, n = bundle.stats("cytosol", "c"), bundle.stats("Nucleus", "n")
    # the realized compartments (VCell's stored volumes: cytosol 14891.9, Nucleus 3697.0 µm³; h = 2 µm)
    assert c[0, 1] / c[0, 0] == pytest.approx(14891.9, rel=0.1)
    assert n[-1, 1] / n[-1, 0] == pytest.approx(3697.0, rel=0.1)
    # the permeability flux moves substance into the nucleus and conserves it
    total = c[:, 1] + n[:, 1]
    assert np.allclose(total, total[0], rtol=1e-10)
    assert n[0, 0] == 0.0 and n[1, 0] > 0.0 and n[2, 0] > n[1, 0]
    # at this coarse h the nucleus reaches ec through the thin cytosol; the run log says so
    assert "'ec' and 'Nucleus' touch but no surface class" in capfd.readouterr().err


def test_several_species_per_compartment_are_refused(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    # VCell's tutorial Ran model: three species in the cytosol — beyond the two-compartment solver, which
    # couples one per compartment; refused before any meshing, with the species named
    argv = ["--math", str(_FIXTURES / "ran_math.yaml"), "--geometry", str(_GEOMETRY), "--t-final", "0.1"]
    assert main([*argv, "--h", "2.0", "--out", str(tmp_path)]) != 0
    assert "'cytosol' has 3 (Ran_cyt, C_cyt, RanC_cyt)" in capfd.readouterr().err
