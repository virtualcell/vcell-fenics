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
from typing import Any

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
    # the VCell compartment names, and the membrane between them -- no species of its own, but written with
    # its maps onto both sides, so a membrane function of the two concentrations can be drawn on it
    assert set(bundle.manifest.domains) == {"cyto_dom", "ext_dom", "mem_dom"}
    membrane = bundle.manifest.domains["mem_dom"]
    assert membrane.kind == "membrane" and membrane.adjacent is not None
    assert set(membrane.adjacent.compartments) == set(membrane.adjacent.maps) == {"cyto_dom", "ext_dom"}
    assert not [v for v in bundle.manifest.variables if v.domain == "mem_dom"]
    for compartment in ("cyto_dom", "ext_dom"):
        mapped = bundle.adjacent("mem_dom", compartment)
        assert mapped is not None and (mapped >= 0).all()
        assert np.array_equal(bundle.mesh(compartment).points[mapped], bundle.mesh("mem_dom").points)
    assert bundle.times == pytest.approx((0.0, 0.05, 0.1, 0.15, 0.2))  # every output, incl. the IC

    summary = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())
    assert summary["run"]["backend"] == "multi_compartment"
    assert sorted(summary["species"]) == ["s_cyto", "s_ext"]
    species = summary["outputs"][-1]["species"]
    assert species["s_ext"]["total"] > 0.0  # substance actually crossed the membrane
    # s_cyto starts at 1 throughout the cytosol and s_ext at 0, so the initial total substance is
    # the *realized* cytosol area (recoverable as total/mean — using the realized area, not the
    # analytic πr², isolates solver conservation from the faceted-circle geometry error). Nothing
    # leaves the pair: the walls are no-flux and the only transport is the membrane flux.
    cytosol_area = species["s_cyto"]["total"] / species["s_cyto"]["mean"]
    assert species["s_cyto"]["total"] + species["s_ext"]["total"] == pytest.approx(cytosol_area, rel=1e-6)


def test_run_couples_several_species_per_compartment(tmp_path: Path) -> None:
    # The VCell permeability model with a second species in each compartment (vcell-fenics #183): in the
    # cytosol p_cyto is made from s_cyto by a reaction (so the two are coupled), and the exterior carries
    # an independent q_ext. The two-compartment solver took one species each and refused this.
    math = yaml.safe_load((_CV / "coupled_perm_math.yaml").read_text())
    math["constants"] += [
        {"name": "kconv", "exp": "2.0"},
        {"name": "p_cyto_diffusionRate", "exp": "0.5"},
        {"name": "q_ext_diffusionRate", "exp": "1.0"},
        {"name": "q_ext_init_uM", "exp": "0.25"},
    ]
    math["variables"] += [
        {"name": "p_cyto", "var_type": "VolumeVariable", "domain": "cyto_dom"},
        {"name": "q_ext", "var_type": "VolumeVariable", "domain": "ext_dom"},
    ]
    cyto, ext = math["compartment_subdomains"]
    cyto["pde_equations"][0]["rate"] = "(- kconv * s_cyto)"
    zero_flux = {side: "0.0" for side in ("xm", "xp", "ym", "yp", "zm", "zp")}
    cyto["pde_equations"].append(
        {
            "name": "p_cyto",
            "rate": "(kconv * s_cyto)",
            "diffusion": "p_cyto_diffusionRate",
            "initial": "0.0",
            "solution_type": "Unknown",
            "boundaries": zero_flux,
        }
    )
    ext["pde_equations"].append(
        {
            "name": "q_ext",
            "rate": "0.0",
            "diffusion": "q_ext_diffusionRate",
            "initial": "q_ext_init_uM",
            "solution_type": "Unknown",
            "boundaries": zero_flux,
        }
    )
    math_file = tmp_path / "multi_math.yaml"
    math_file.write_text(yaml.safe_dump(math, sort_keys=False))
    out = tmp_path / "results"
    argv = ["--math", str(math_file), "--geometry", str(_CV / "coupled_perm_geom.yaml"), "--t-final", "0.4"]
    status = main([*argv, "--output-dt", "0.1", "--h", "0.12", "--out", str(out)])
    assert status == 0
    bundle = Bundle.open(out / "results.fenics")
    assert list(bundle.manifest.domains) == ["cyto_dom", "ext_dom", "mem_dom"]  # the species-less membrane last
    summary = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())
    assert summary["run"]["backend"] == "multi_compartment"
    assert sorted(summary["species"]) == ["p_cyto", "q_ext", "s_cyto", "s_ext"]
    rows = [row["species"] for row in summary["outputs"]]
    assert len(rows) == 5 and rows[-1]["p_cyto"]["total"] > 0.0  # the reaction ran
    # s moves across the membrane and converts to p, but never leaves: s_cyto + p_cyto + s_ext is conserved,
    # and q_ext (untouched) keeps its total
    linked = [r["s_cyto"]["total"] + r["p_cyto"]["total"] + r["s_ext"]["total"] for r in rows]
    assert max(linked) - min(linked) <= 1e-11 * linked[0]
    q = [r["q_ext"]["total"] for r in rows]
    assert max(q) - min(q) <= 1e-12 * q[0]


def test_run_couples_both_compartments_and_membrane_species(tmp_path: Path) -> None:
    # A VCell receptor model: ligand in both compartments (s_cyto | s_ext) bound by a membrane receptor R.
    # Equations on three subdomains — the two compartments and the membrane between them — route to the
    # membrane-coupled solver (vcell-fenics #183) instead of being refused, with the membrane recorded as
    # a domain of its own, and the substance conserved across the binding.
    out = tmp_path / "results"
    status = main(
        [
            "--math",
            str(_CV / "receptor_math.yaml"),
            "--geometry",
            str(_CV / "receptor_geom.yaml"),
            "--t-final",
            "0.5",
            "--output-dt",
            "0.25",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    bundle = Bundle.open(out / "results.fenics")
    assert set(bundle.manifest.domains) == {"cyto_dom", "ext_dom", "mem_dom"}
    assert bundle.manifest.domains["mem_dom"].kind == "membrane"
    assert bundle.times == pytest.approx((0.0, 0.25, 0.5))  # every output, incl. the IC

    summary = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())
    assert summary["run"]["backend"] == "multi_compartment"
    assert sorted(summary["species"]) == ["R", "s_cyto", "s_ext"]
    # Free ligand (µM, over the compartments) plus bound receptor (molecules/µm², over the membrane)
    # reconciled by KMOLE is the conserved substance: flat to round-off, while R actually binds.
    constants = yaml.safe_load((_CV / "receptor_math.yaml").read_text())["constants"]
    kmole = next(float(c["exp"]) for c in constants if c["name"] == "KMOLE")
    totals = [
        row["species"]["s_cyto"]["total"] + row["species"]["s_ext"]["total"] + kmole * row["species"]["R"]["total"]
        for row in summary["outputs"]
    ]
    assert summary["outputs"][-1]["species"]["R"]["total"] > 0.0
    assert max(totals) - min(totals) <= 1e-12 * totals[0]


def _receptor(tmp_path: Path, name: str, edit: Any) -> list[str]:
    """The receptor model (ligand in both compartments binding a membrane receptor R), edited by ``edit``,
    as CLI arguments."""
    math = yaml.safe_load((_CV / "receptor_math.yaml").read_text())
    edit(math)
    math_file = tmp_path / f"{name}_math.yaml"
    math_file.write_text(yaml.safe_dump(math, sort_keys=False))
    return ["--math", str(math_file), "--geometry", str(_CV / "receptor_geom.yaml"), "--t-final", "0.5"]


def _no_cytosolic_binding(math: dict[str, Any]) -> None:
    for function in math["functions"]:
        if function["name"] == "J_bind_in":
            function["exp"] = "0.0"  # nothing binds from the cytosol


def _one_sided(math: dict[str, Any]) -> None:
    """Drop the cytosolic ligand: membrane species with bulk species in the extracellular space only."""
    _no_cytosolic_binding(math)
    math["variables"] = [v for v in math["variables"] if v["name"] != "s_cyto"]
    for compartment in math["compartment_subdomains"]:
        if compartment["name"] == "cyto_dom":
            compartment["pde_equations"] = []
    for membrane in math["membrane_subdomains"]:
        membrane["jump_conditions"] = [j for j in membrane.get("jump_conditions", []) if j["name"] != "s_cyto"]


def test_membrane_species_with_one_sided_bulk_run(tmp_path: Path) -> None:
    # The receptor model without its cytosolic ligand: membrane species with bulk species on one side only
    # (#185). The empty cytosol carries no unknowns, so the run must equal the same model WITH a cytosolic ligand
    # that binds nothing, and it conserves the ligand: the free
    # extracellular pool plus the bound receptors, KMOLE-reconciled (∫s_ext dV + KMOLE·∫R dA).
    one_sided, inert = tmp_path / "one_sided", tmp_path / "inert"
    options = ["--output-dt", "0.25", "--h", "0.3", "--no-fields"]
    assert main([*_receptor(tmp_path, "one_sided", _one_sided), *options, "--out", str(one_sided)]) == 0
    assert main([*_receptor(tmp_path, "inert", _no_cytosolic_binding), *options, "--out", str(inert)]) == 0
    a, b = Bundle.open(one_sided / "results.fenics"), Bundle.open(inert / "results.fenics")
    assert "cyto_dom" not in a.manifest.domains  # nothing to write for the unmodelled side
    for domain, variable in (("ext_dom", "s_ext"), ("mem_dom", "R")):
        # to the adaptive integrator's tolerance: the two solves have different blocks, so different steps
        assert np.allclose(a.stats(domain, variable), b.stats(domain, variable), rtol=1e-5, atol=1e-8), variable
    kmole = 1.0 / 602.214076
    total = a.stats("ext_dom", "s_ext")[:, 1] + kmole * a.stats("mem_dom", "R")[:, 1]
    assert np.allclose(total, total[0], rtol=1e-7)
    assert a.stats("mem_dom", "R")[-1, 1] > a.stats("mem_dom", "R")[0, 1]  # it did bind


def test_a_well_mixed_compartment_beside_membrane_species_runs(tmp_path: Path) -> None:
    # A well-mixed cytosolic ligand (VCell's VolumeRegionVariable, §1.4.2 T5) as the cytosol's only species,
    # beside membrane receptors: the cytosol's field block is a placeholder and s_cyto a Real block (#185,
    # #196). One value on the whole cytosol, and the ligand is conserved across both pools and the receptors.
    def well_mixed(math: dict[str, Any]) -> None:
        for variable in math["variables"]:
            if variable["name"] == "s_cyto":
                variable["var_type"] = "VolumeRegionVariable"
        for compartment in math["compartment_subdomains"]:
            if compartment["name"] == "cyto_dom":
                compartment["pde_equations"] = [p for p in compartment["pde_equations"] if p["name"] != "s_cyto"]
                compartment["volume_region_equations"] = [
                    {"name": "s_cyto", "uniform_rate": "0.0", "volume_rate": "0.0", "initial": "s_cyto_init_uM"}
                ]

    out = tmp_path / "out"
    options = ["--output-dt", "0.25", "--h", "0.3", "--no-fields", "--out", str(out)]
    assert main([*_receptor(tmp_path, "well_mixed", well_mixed), *options]) == 0
    bundle = Bundle.open(out / "results.fenics")
    s_cyto = bundle.stats("cyto_dom", "s_cyto")
    assert np.allclose(s_cyto[:, 2], s_cyto[:, 3])  # one value on the whole cytosol
    kmole = 1.0 / 602.214076
    total = s_cyto[:, 1] + bundle.stats("ext_dom", "s_ext")[:, 1] + kmole * bundle.stats("mem_dom", "R")[:, 1]
    assert np.allclose(total, total[0], rtol=1e-7)


def test_well_mixed_species_and_membrane_potential_run_on_the_two_compartment_solver(tmp_path: Path) -> None:
    # The permeability model with its cytosolic species well-mixed (a VolumeRegionVariable) and a membrane
    # potential added (a MembraneRegionVariable, C dV/dt = −g (V − E)): the two-compartment solver hosts both
    # as Real blocks (#196). The bundle holds s_cyto as a constant field on the cytosol, V on the membrane.
    math = yaml.safe_load((_CV / "coupled_perm_math.yaml").read_text())
    for variable in math["variables"]:
        if variable["name"] == "s_cyto":
            variable["var_type"] = "VolumeRegionVariable"
    math["variables"].append({"name": "V", "var_type": "MembraneRegionVariable", "domain": "mem_dom"})
    math["constants"] += [
        {"name": "g_leak", "exp": "2.0"},
        {"name": "C_m", "exp": "0.5"},
        {"name": "E_rev", "exp": "-60.0"},
    ]
    for compartment in math["compartment_subdomains"]:
        if compartment["name"] == "cyto_dom":
            compartment["pde_equations"] = []
            compartment["volume_region_equations"] = [
                {"name": "s_cyto", "uniform_rate": "0.0", "volume_rate": "0.0", "initial": "s_cyto_init_uM"}
            ]
    for membrane in math["membrane_subdomains"]:
        membrane["membrane_region_equations"] = [
            {"name": "V", "uniform_rate": "0.0", "membrane_rate": "(-g_leak * (V - E_rev) / C_m)", "initial": "-20.0"}
        ]
    math_file = tmp_path / "region_math.yaml"
    math_file.write_text(yaml.safe_dump(math, sort_keys=False))
    out = tmp_path / "out"
    argv = ["--math", str(math_file), "--geometry", str(_CV / "coupled_perm_geom.yaml"), "--t-final", "1.0"]
    assert main([*argv, "--output-dt", "0.5", "--h", "0.2", "--no-fields", "--out", str(out)]) == 0

    bundle = Bundle.open(out / "results.fenics")
    s_cyto = bundle.stats("cyto_dom", "s_cyto")  # (T, 4): mean, total, min, max
    assert np.allclose(s_cyto[:, 2], s_cyto[:, 3])  # one value on the whole cytosol
    total = s_cyto[:, 1] + bundle.stats("ext_dom", "s_ext")[:, 1]
    assert np.allclose(total, total[0], rtol=1e-8)  # exchange only: mass conserved
    potential = bundle.stats("mem_dom", "V")[:, 0]
    exact = -60.0 + 40.0 * np.exp(-4.0 * np.array([0.0, 0.5, 1.0]))
    assert np.allclose(potential, exact, atol=1e-2)


def test_a_membrane_potential_runs_beside_membrane_receptors(tmp_path: Path) -> None:
    # The receptor model (ligand in both compartments, receptors on the membrane) with a membrane potential
    # added, C dV/dt = −g (V − E): the membrane-coupled solver hosts it as a Real block (#196), written on
    # the membrane domain as a constant field beside the receptor density.
    math = yaml.safe_load((_CV / "receptor_math.yaml").read_text())
    membrane = math["membrane_subdomains"][0]["name"]
    math["variables"].append({"name": "V", "var_type": "MembraneRegionVariable", "domain": membrane})
    math["constants"] += [
        {"name": "g_leak", "exp": "2.0"},
        {"name": "C_m", "exp": "0.5"},
        {"name": "E_rev", "exp": "-60.0"},
    ]
    math["membrane_subdomains"][0]["membrane_region_equations"] = [
        {"name": "V", "uniform_rate": "0.0", "membrane_rate": "(-g_leak * (V - E_rev) / C_m)", "initial": "-20.0"}
    ]
    math_file = tmp_path / "receptor_v_math.yaml"
    math_file.write_text(yaml.safe_dump(math, sort_keys=False))
    out = tmp_path / "out"
    argv = ["--math", str(math_file), "--geometry", str(_CV / "receptor_geom.yaml"), "--t-final", "1.0"]
    assert main([*argv, "--output-dt", "0.5", "--h", "0.25", "--no-fields", "--out", str(out)]) == 0

    bundle = Bundle.open(out / "results.fenics")
    potential = bundle.stats(membrane, "V")  # (T, 4): mean, total, min, max
    assert np.allclose(potential[:, 2], potential[:, 3])  # one value on the whole membrane
    exact = -60.0 + 40.0 * np.exp(-4.0 * np.array([0.0, 0.5, 1.0]))
    assert np.allclose(potential[:, 0], exact, atol=1e-2)


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


def test_a_vcml_application_with_a_fast_system_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # the .vcml path reads the FastSystem from the XML (pyvcell drops it) and refuses it before solving
    pytest.importorskip("pyvcell.vcml.vcml_reader")
    text = (_CV / "minimal_diffusion_2d_v1.vcml").read_text()
    marker = "</CompartmentSubDomain>"
    assert text.count(marker) == 1
    fast = "<FastSystem><FastInvariant>(u + uB)</FastInvariant><FastRate>J_buffering</FastRate></FastSystem>"
    model = tmp_path / "fast.vcml"
    model.write_text(text.replace(marker, fast + marker))
    assert main(["--vcml", str(model), "--out", str(tmp_path / "out")]) == 2
    assert "FastSystem" in capsys.readouterr().err
