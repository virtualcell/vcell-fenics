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

import numpy as np
import pytest
import yaml

from vcell_fenics.cli import CliError, build_parser, detect_format, load_pair, main, resolve_options
from vcell_fenics.results import Bundle

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
    assert options.t_final == 1.0
    assert options.output_times == pytest.approx([0.1 * k for k in range(11)])
    assert options.h == pytest.approx(2.0 / 128.0)  # extent 2.0 over the FV grid's 128 cells


def test_explicit_flags_beat_the_model() -> None:
    model = load_pair(_MATH, _GEOM)
    options = resolve_options(model, _args(t_final=2.0, output_dt=0.5, dt=0.25, h=0.2))
    assert (options.t_final, options.dt, options.h) == (2.0, 0.25, 0.2)
    assert options.output_times == (0.0, 0.5, 1.0, 1.5, 2.0)


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
    # One self-contained bundle per run (ADR 010): the mesh, the fields at every output time, their
    # statistics, the manifest — and the resolved formalism + summary as provenance.
    bundle_dir = out / "results.fenics"
    for name in ("mesh/domain.vtu", "domain/u", "domain/v", "provenance/math.yaml", "provenance/summary.json"):
        assert (bundle_dir / name).exists(), name

    bundle = Bundle.open(bundle_dir)
    assert bundle.status == "completed"
    assert bundle.times == pytest.approx((0.0, 0.1, 0.2))
    assert [(v.domain, v.name) for v in bundle.manifest.variables] == [("domain", "u"), ("domain", "v")]
    assert bundle.series("domain", "u").shape == (3, bundle.manifest.domains["domain"].n_points)

    summary = json.loads((bundle_dir / "provenance" / "summary.json").read_text())
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
    # The bundle's statistics are the same numbers.
    assert bundle.stats("domain", "u")[-1][1] == pytest.approx(last["u"]["total"])


def test_method_of_lines_writes_every_output_time(tmp_path: Path) -> None:
    """The adaptive integrator used to record t = 0 and t_final only; VCell needs every output time."""

    out = tmp_path / "results"
    argv = ["--math", str(_MATH), "--geometry", str(_GEOM), "--t-final", "0.4", "--output-dt", "0.1"]
    status = main([*argv, "--h", "0.25", "--time-integration", "method_of_lines", "--out", str(out)])
    assert status == 0
    bundle = Bundle.open(out / "results.fenics")
    assert bundle.times == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4))
    totals = bundle.stats("domain", "u")[:, 1]
    # u decays at k_conv = 0.5 (diffusion conserves it behind no-flux walls): the adaptive run
    # tracks e^{-k t} at every output, not just the last.
    assert totals == pytest.approx(totals[0] * np.exp(-0.5 * np.array(bundle.times)), rel=1e-4)


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
    bundle_dir = out / "results.fenics"
    assert (bundle_dir / "mesh").is_dir() and (bundle_dir / "stats").is_dir()
    assert not any((bundle_dir / domain).exists() for domain in Bundle.open(bundle_dir).manifest.domains)  # --no-fields

    provenance = bundle_dir / "provenance"
    model = load_pair(provenance / "math.yaml", provenance / "geometry.yaml")
    assert model.source == "native"
    assert model.geometry.name == "square"
    assert [v.name for v in model.math.variables] == ["u"]

    summary = json.loads((provenance / "summary.json").read_text())
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
            "--output-dt",
            "0.05",
            "--h",
            "0.12",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    bundle = Bundle.open(out / "results.fenics")
    assert set(bundle.manifest.domains) == {"cyto_dom", "ext_dom"}  # the VCell compartment names
    assert bundle.times == pytest.approx((0.0, 0.05, 0.1, 0.15, 0.2))  # every output, incl. the IC

    summary = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())
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


def test_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    """argparse %-formats help strings: a bare '%' (as in the [[[progress:…%]]] marker) crashes --help."""

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--simtask" in out and "[[[progress:…%]]]" in out
