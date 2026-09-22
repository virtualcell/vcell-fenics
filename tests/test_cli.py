"""The command-line runner (`vcell_fenics.cli`) — the entry point the container image wraps.

Covers the parts that are the CLI's own responsibility, not the solver's: telling the two
formalisms apart, merging run settings from flags / the VCell simulation / defaults, and
producing the output tree a bind-mounted results directory is expected to contain.

The end-to-end cases run the *native* example model (`examples/models/`) at a deliberately
coarse `h` so the whole file stays a few seconds; the physics behind them is verified in
`tests/test_backend_*` and `mms/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from vcell_fenics.cli import CliError, build_parser, detect_format, load_pair, main, resolve_options

_ROOT = Path(__file__).resolve().parent.parent
_MATH = _ROOT / "examples" / "models" / "diffusion2d_math.yaml"
_GEOM = _ROOT / "examples" / "models" / "diffusion2d_geom.yaml"
_CV = _ROOT / "cross_validation"


# --- format detection ---------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        (_MATH, "native"),
        (_GEOM, "native"),
        (_CV / "minimal_diffusion_2d_v1_math.yaml", "vcell"),
        (_CV / "minimal_diffusion_2d_v1_geom.yaml", "vcell"),
    ],
)
def test_detect_format(path: Path, expected: str) -> None:
    assert detect_format(yaml.safe_load(path.read_text())) == expected


def test_detect_format_rejects_ambiguous() -> None:
    with pytest.raises(CliError, match="cannot tell whether"):
        detect_format({"name": "something", "dim": 2})


def test_load_pair_rejects_mixed_formalisms() -> None:
    with pytest.raises(CliError, match="must be in one formalism"):
        load_pair(_MATH, _CV / "minimal_diffusion_2d_v1_geom.yaml")


def test_load_pair_rejects_geometry_name_mismatch(tmp_path: Path) -> None:
    # The geometry *name* is the contract between the two documents (§3.4 name resolution);
    # a mismatch means the math would be solved on a domain nobody asked for.
    geom = yaml.safe_load(_GEOM.read_text())
    geom["geometry_description"]["name"] = "not_square"
    renamed = tmp_path / "geom.yaml"
    renamed.write_text(yaml.safe_dump(geom))
    with pytest.raises(CliError, match="binds geometry 'square'"):
        load_pair(_MATH, renamed)


# --- option resolution --------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    args = build_parser().parse_args(["--math", str(_MATH), "--geometry", str(_GEOM)])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_options_require_t_final_when_the_model_has_none() -> None:
    # A native model carries no duration, so there is nothing to default to.
    with pytest.raises(CliError, match="--t-final is required"):
        resolve_options(load_pair(_MATH, _GEOM), _args())


def test_options_default_from_the_vcell_simulation() -> None:
    # A .vcml's simulation supplies duration, output interval, and (via mesh_size) an element
    # size, so `--vcml model.vcml` alone is a complete run request.
    pytest.importorskip("pyvcell.vcml.vcml_reader")
    from vcell_fenics.cli import load_vcml

    model = load_vcml(_CV / "minimal_diffusion_2d_v1.vcml")
    options = resolve_options(model, _args())
    assert (options.t_final, options.output_dt) == (1.0, 0.1)
    assert options.h == pytest.approx(2.0 / 128.0)  # extent 2.0 over the FV grid's 128 cells


def test_explicit_flags_beat_the_model() -> None:
    model = load_pair(_MATH, _GEOM)
    options = resolve_options(model, _args(t_final=2.0, output_dt=0.5, dt=0.25, h=0.2))
    assert (options.t_final, options.output_dt, options.dt, options.h) == (2.0, 0.5, 0.25, 0.2)


def test_h_defaults_to_the_narrowest_axis_when_nothing_says() -> None:
    options = resolve_options(load_pair(_MATH, _GEOM), _args(t_final=1.0))
    assert options.h == pytest.approx(2.0 / 32.0)


# --- end to end ---------------------------------------------------------------


def test_run_writes_the_result_tree(tmp_path: Path) -> None:
    out = tmp_path / "results"
    status = main(
        [
            "--math",
            str(_MATH),
            "--geometry",
            str(_GEOM),
            "--t-final",
            "0.2",
            "--output-dt",
            "0.1",
            "--dt",
            "0.05",
            "--h",
            "0.25",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    # The full tree a mounted results directory is expected to receive: the resolved formalism
    # (provenance), the fields, and the machine-readable summary.
    for name in ("math.yaml", "geometry.yaml", "fields.xdmf", "fields.h5", "summary.json"):
        assert (out / name).is_file(), name

    summary = json.loads((out / "summary.json").read_text())
    assert summary["species"] == ["u", "v"]
    assert summary["run"]["backend"] == "single_mesh"
    assert summary["run"]["steps"] == 4
    assert [record["t"] for record in summary["outputs"]] == pytest.approx([0.0, 0.1, 0.2])

    # u decays into v at k_conv = 0.5, so ∫u dx falls by ≈ exp(-0.5 · 0.2) while v (initially
    # empty, decaying at 0.2) appears. Backward Euler at dt = 0.05 is a coarse but bounded
    # approximation of that — this is a smoke test of the *pipeline*, so keep the tolerance loose.
    first, last = summary["outputs"][0]["species"], summary["outputs"][-1]["species"]
    assert last["u"]["total"] == pytest.approx(first["u"]["total"] * 0.9048, rel=0.05)
    assert first["v"]["total"] == 0.0
    assert last["v"]["total"] > 0.0


def test_run_writes_the_resolved_formalism_for_a_vcell_model(tmp_path: Path) -> None:
    # The VCell path's real payload: what a .vcml turned into. It must come back as a *native*
    # document, round-trippable by the native loader — that is what makes the container's output
    # directory a debuggable record of the import.
    pytest.importorskip("pyvcell.vcml.vcml_reader")
    out = tmp_path / "results"
    status = main(
        [
            "--vcml",
            str(_CV / "minimal_diffusion_2d_v1.vcml"),
            "--t-final",
            "0.1",
            "--dt",
            "0.05",
            "--h",
            "0.3",
            "--out",
            str(out),
            "--no-fields",
        ]
    )
    assert status == 0
    assert not (out / "fields.xdmf").exists()  # --no-fields

    model = load_pair(out / "math.yaml", out / "geometry.yaml")
    assert model.source == "native"
    assert model.geometry.name == "square"
    assert [v.name for v in model.math.variables] == ["u"]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["source"]["biomodel"] == "MinimalDiffusion2D"
    assert summary["source"]["application"] == "diffusion2D"
    # Pure diffusion behind no-flux walls: the total is conserved.
    totals = [record["species"]["u"]["total"] for record in summary["outputs"]]
    assert totals[-1] == pytest.approx(totals[0], rel=1e-6)


def test_run_couples_two_compartments_across_a_membrane(tmp_path: Path) -> None:
    # A VCell permeability model (s_cyto | membrane | s_ext) must route to the interface-coupled
    # solver rather than being rejected — and conserve total substance across the membrane.
    out = tmp_path / "results"
    status = main(
        [
            "--math",
            str(_CV / "coupled_perm_math.yaml"),
            "--geometry",
            str(_CV / "coupled_perm_geom.yaml"),
            "--t-final",
            "0.2",
            "--h",
            "0.12",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    assert (out / "inner.xdmf").is_file() and (out / "outer.xdmf").is_file()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["backend"] == "interface_coupled"
    assert sorted(summary["species"]) == ["s_cyto", "s_ext"]
    species = summary["outputs"][-1]["species"]
    assert species["s_ext"]["total"] > 0.0  # substance actually crossed the membrane
    # s_cyto starts at 1 throughout the cytosol and s_ext at 0, so the initial total substance is
    # the *realized* cytosol area (recoverable as total/mean — using the realized area, not the
    # analytic πr², isolates solver conservation from the faceted-circle geometry error). Nothing
    # leaves the pair: the walls are no-flux and the only transport is the membrane flux.
    cytosol_area = species["s_cyto"]["total"] / species["s_cyto"]["mean"]
    assert species["s_cyto"]["total"] + species["s_ext"]["total"] == pytest.approx(cytosol_area, rel=1e-6)


def test_unknown_model_file_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--math", "nope.yaml", "--geometry", str(_GEOM), "--t-final", "1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_model_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "give a model" in capsys.readouterr().err


def test_nonlinear_model_under_backward_euler_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real VCell kinetics are routinely nonlinear, which backward Euler cannot lower. That is a model /
    option mismatch the user can fix (switch to method of lines), so it exits 2 with the named fix —
    not a traceback."""

    document = yaml.safe_load(_MATH.read_text())
    document["math_description"]["equations"][0]["terms"]["source"] = "-k_conv * u * u"
    math = tmp_path / "nonlinear_math.yaml"
    math.write_text(yaml.safe_dump(document))
    argv = ["--math", str(math), "--geometry", str(_GEOM), "--t-final", "0.1", "--h", "0.5"]
    assert main([*argv, "--time-integration", "backward_euler", "--out", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "NonlinearTermError" in err
    assert "method-of-lines" in err
