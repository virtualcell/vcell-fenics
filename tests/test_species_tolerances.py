"""Error tolerances that mean the same thing for every species (ADR 022, `backend/tolerances.py`).

Two kinds of test. The first integrate small systems of ODEs with PETSc's adaptive BDF directly, where the exact
solution is known and every decision can be isolated: a species' scale (decision 1), the balance of the norm
(decision 2), a held unknown (decision 3). The others run the real integrators and check that each one lays out
its species as the helper expects, and that a species twelve decades below another is controlled in a coupled
finite-element solve.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.tolerances import (
    SpeciesScales,
    ToleranceControl,
    ToleranceScaling,
    blocked_layout,
    tolerance_summary,
)
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_Rate = Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]

# --- small ODE systems ------------------------------------------------------------------------------------------


def _integrate(
    u0: npt.NDArray[np.float64],
    rate: _Rate,
    layout: dict[str, npt.NDArray[np.int64]],
    t_final: float,
    *,
    scaling: ToleranceScaling,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    held: npt.NDArray[np.int64] | None = None,
    scales: SpeciesScales | None = None,
    stops: tuple[float, ...] = (),
) -> tuple[npt.NDArray[np.float64], int, SpeciesScales]:
    """``u' = rate(t, u)`` with adaptive BDF from ``u0`` to ``t_final``, a fresh TS at each of ``stops`` (a stride
    boundary) sharing one `ToleranceControl`. Returns the final state, the steps and the scales."""
    n = len(u0)
    x = PETSc.Vec().createSeq(n)
    x.array[:] = u0
    eps = 1.0e-7

    def residual(_ts: PETSc.TS, t: float, u: PETSc.Vec, u_dot: PETSc.Vec, f: PETSc.Vec) -> None:
        f.array[:] = u_dot.array_r - rate(t, _real(u))

    def jacobian(
        _ts: PETSc.TS, t: float, u: PETSc.Vec, _u_dot: PETSc.Vec, shift: float, a: PETSc.Mat, p: PETSc.Mat
    ) -> None:
        base = _real(u)
        f0 = rate(t, base)
        dense = np.zeros((n, n))
        for j in range(n):
            step = eps * max(1.0e-30, abs(base[j]))
            bumped = base.copy()
            bumped[j] += step
            dense[:, j] = -(rate(t, bumped) - f0) / step
        dense += shift * np.eye(n)
        p.zeroEntries()
        p.setValues(range(n), range(n), dense.ravel().tolist())
        p.assemble()

    control = ToleranceControl(
        x,
        layout,
        rtol=rtol,
        atol=atol,
        scaling=scaling,
        scales=scales,
        held=held,
    )
    steps, t = 0, 0.0
    for t_end in (*stops, t_final):
        matrix = PETSc.Mat().createDense((n, n))
        matrix.setUp()
        ts = PETSc.TS().create()
        ts.setType("bdf")
        ts.setIFunction(residual, x.duplicate())
        ts.setIJacobian(jacobian, matrix, matrix)
        ts.setTime(t)
        ts.setTimeStep(min(1.0e-4, t_end - t))
        ts.setMaxTime(t_end)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
        control.attach(ts)
        ts.solve(x)
        steps += ts.getStepNumber()
        ts.destroy()
        matrix.destroy()
        t = t_end
    out = _real(x)
    control.destroy()
    x.destroy()
    return out, steps, control.scales


def _real(x: PETSc.Vec) -> npt.NDArray[np.float64]:
    return np.array(x.array_r.real, dtype=np.float64)


_SCALINGS: tuple[ToleranceScaling, ...] = ("species", "scalar")


def _two(n_a: int, n_b: int) -> dict[str, npt.NDArray[np.int64]]:
    return {"a": np.arange(n_a), "b": np.arange(n_a, n_a + n_b)}


def _decay(k: npt.NDArray[np.float64]) -> _Rate:
    return lambda _t, u: -k * u


# --- decision 1: each species' absolute tolerance from its own scale ---------------------------------------------


def test_a_species_atol_is_the_smaller_of_atol_and_rtol_times_its_scale() -> None:
    scales = SpeciesScales({"big": 10.0, "small": 1.0e-9, "zero": 0.0, "noise": 1.0e-17})
    atols = scales.species_atol(rtol=1.0e-6, atol=1.0e-8)
    assert atols["big"] == 1.0e-8  # rtol·10 = 1e-5 > atol: the ceiling
    assert atols["small"] == pytest.approx(1.0e-15)
    assert atols["zero"] == 1.0e-8  # nothing to scale by
    assert atols["noise"] == 1.0e-8  # below ε·10 ≈ 2e-15: the round-off floor


def test_the_error_is_the_same_at_every_scale() -> None:
    """The scale table of ADR 022: one species ``c' = −c`` at sizes 1, 1e-6 and 1e-12, beside an O(1) species.
    With one scalar atol the small ones are not controlled; with their own, every size errs alike, within the
    factor decision 1's limitation leaves: a species is held to ``rtol·peak`` absolute as it decays, where the O(1)
    one meets the tighter ceiling ``atol`` (measured 1.5e-4 at size 1, 4.2e-4 below; scalar 1.8e-2 below)."""
    k = np.array([0.1, 0.1, 1.0, 1.0])  # the small species' own dynamics drive its error, not the O(1) one's steps
    errors: dict[ToleranceScaling, list[float]] = {"species": [], "scalar": []}
    for scale in (1.0, 1.0e-6, 1.0e-12):
        u0 = np.array([1.0, 1.0, scale, scale * 0.5])
        exact = u0 * np.exp(-k * 2.0)
        for scaling in errors:
            u, _, _ = _integrate(u0, _decay(k), _two(2, 2), 2.0, scaling=scaling)
            errors[scaling].append(float(np.max(np.abs(u[2:] - exact[2:]) / exact[2:])))
    assert max(errors["species"]) < 3.0 * min(errors["species"]), errors["species"]
    assert errors["species"][1] == pytest.approx(errors["species"][2], rel=1.0e-6)  # scale-free below the ceiling
    assert errors["scalar"][2] > 30.0 * errors["species"][2], errors  # 1e-12 against atol 1e-8: unchecked


def test_a_species_that_starts_at_zero_is_controlled_at_the_size_it_grows_to() -> None:
    """``b' = 1e-6·(1 − b/1e-6)·3`` from zero: the running maximum follows it, and survives a stride boundary."""
    k = np.array([0.1, 3.0])

    def rate(_t: float, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([-k[0] * u[0], k[1] * (1.0e-6 - u[1])])

    exact = 1.0e-6 * (1.0 - math.exp(-3.0 * 2.0))
    u_species, _, scales = _integrate(np.array([1.0, 0.0]), rate, _two(1, 1), 2.0, scaling="species", stops=(0.7,))
    u_scalar, _, _ = _integrate(np.array([1.0, 0.0]), rate, _two(1, 1), 2.0, scaling="scalar", stops=(0.7,))
    assert scales.peak["b"] == pytest.approx(u_species[1], rel=1.0e-12)  # raised every step, kept over the stop
    assert scales.peak["a"] == 1.0  # a decay keeps its initial peak
    error_species = abs(u_species[1] - exact) / exact
    error_scalar = abs(u_scalar[1] - exact) / exact
    assert error_species < 1.0e-4
    assert error_scalar > 10.0 * error_species


def test_a_round_off_sized_species_does_not_collapse_the_step() -> None:
    """A species at 1e-30 beside an O(1) one keeps the scalar atol (the floor): exactly the steps it takes at zero.
    Without the floor its atol would be 1e-36 and the step would collapse."""
    k = np.array([1.0, 2.0])
    _, steps_tiny, scales = _integrate(np.array([1.0, 1.0e-30]), _decay(k), _two(1, 1), 2.0, scaling="species")
    _, steps_zero, _ = _integrate(np.array([1.0, 0.0]), _decay(k), _two(1, 1), 2.0, scaling="species")
    assert scales.species_atol(1.0e-6, 1.0e-8)["b"] == 1.0e-8
    assert steps_tiny == steps_zero


def test_a_raised_floor_leaves_a_trace_species_to_the_scalar_atol() -> None:
    """The floor is a solver option (``--tolerance-floor``). A trace species at 1e-9 of the largest is controlled to
    its own scale by default, which costs steps; with the floor at 1e-6 it keeps ``atol`` and takes the steps it
    would take at zero, while a species above the floor keeps its own tolerance."""
    k = np.array([1.0, 2.0])
    u0 = np.array([1.0, 1.0e-9])
    _, steps_default, _ = _integrate(u0, _decay(k), _two(1, 1), 2.0, scaling="species")
    raised = SpeciesScales(floor=1.0e-6)
    _, steps_raised, _ = _integrate(u0, _decay(k), _two(1, 1), 2.0, scaling="species", scales=raised)
    _, steps_zero, _ = _integrate(np.array([1.0, 0.0]), _decay(k), _two(1, 1), 2.0, scaling="species")
    assert steps_default > steps_raised == steps_zero
    assert raised.species_atol(1.0e-6, 1.0e-8)["b"] == 1.0e-8
    assert SpeciesScales({"a": 1.0, "b": 1.0e-5}, floor=1.0e-6).species_atol(1.0e-6, 1.0e-8)["b"] == pytest.approx(
        1.0e-11
    )
    assert tolerance_summary(1.0e-6, 1.0e-8, "species", raised)["tolerance_floor"] == 1.0e-6


@pytest.mark.parametrize("floor", [-1.0e-6, 1.0, math.nan])
def test_a_floor_outside_zero_to_one_is_refused(floor: float) -> None:
    with pytest.raises(ValueError, match="tolerance floor"):
        SpeciesScales(floor=floor)


# --- decision 2: the norm weighs each species equally ------------------------------------------------------------


def test_balanced_weights_are_the_root_of_each_species_share() -> None:
    x = PETSc.Vec().createSeq(10)
    x.set(1.0)
    control = ToleranceControl(x, _two(9, 1), rtol=1.0e-6, atol=1.0e-8, scaling="species")
    assert control.counts == {"a": 9, "b": 1}
    assert control.weights["a"] == pytest.approx(math.sqrt(0.9))
    assert control.weights["b"] == pytest.approx(math.sqrt(0.1))
    control.destroy()
    x.destroy()


def test_a_species_with_one_unknown_keeps_its_control_however_many_the_others_have() -> None:
    """A region variable (one unknown) beside a bulk species of n unknowns: under one plain 2-norm its share of the
    mean square falls as 1/n; balanced, its error is the same whether the bulk has 4 unknowns or 400."""
    errors = []
    for n in (4, 400):
        k = np.concatenate([np.full(n, 0.01), [3.0]])  # the bulk barely moves; the region variable drives the step
        u0 = np.ones(n + 1)
        exact = np.exp(-3.0)
        u, _, _ = _integrate(u0, _decay(k), _two(n, 1), 1.0, scaling="species")
        errors.append(abs(u[-1] - exact) / exact)
    assert errors[1] < 2.0 * errors[0], errors


# --- decision 3: a held unknown stays out of the norm -------------------------------------------------------------


def test_a_held_unknown_does_not_change_the_steps() -> None:
    """A species with one unknown held constant (a Dirichlet node) takes exactly the steps it takes without it."""
    k = np.array([1.0, 1.0, 1.0])
    u0 = np.array([1.0, 0.7, 0.4])
    _, alone, _ = _integrate(u0[:2], _decay(k[:2]), {"a": np.arange(2)}, 2.0, scaling="species")

    def rate(_t: float, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = -k * u
        out[2] = 0.0  # held
        return out

    _, with_held, _ = _integrate(
        u0, rate, {"a": np.arange(3)}, 2.0, scaling="species", held=np.array([2], dtype=np.int64)
    )
    assert with_held == alone


# --- the summary --------------------------------------------------------------------------------------------------


def test_the_summary_records_each_species_tolerance() -> None:
    scales = SpeciesScales({"c": 2.0, "e": 1.0e-12})
    summary = tolerance_summary(1.0e-6, 1.0e-8, "species", scales)
    assert summary["tolerance_scaling"] == "species"
    assert summary["species_tolerances"]["e"] == {"scale": 1.0e-12, "atol": pytest.approx(1.0e-18)}
    assert summary["species_tolerances"]["c"]["atol"] == 1.0e-8
    assert "species_tolerances" not in tolerance_summary(1.0e-6, 1.0e-8, "scalar", scales)


def test_a_blocked_layout_interleaves_components_and_skips_a_placeholder() -> None:
    layout = blocked_layout([(0, 6, 2, ["p", "q"]), (6, 3, 1, []), (9, 1, 1, ["r"])])
    assert layout["p"].tolist() == [0, 2, 4]
    assert layout["q"].tolist() == [1, 3, 5]
    assert layout["r"].tolist() == [9]
    assert set(layout) == {"p", "q", "r"}  # the placeholder block (6..8) is no species


# --- the integrators lay out their species as the helper expects --------------------------------------------------


def _nested_cell() -> GeometryDescription:
    """A nucleus (r = 0.3) in a cytosol (r = 0.7) in a 2 × 2 box of extracellular space."""
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="nuc", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.09"),
            SubVolume(name="cyt", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.49"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="ne", inside="nuc", outside="cyt"),
            SurfaceClass(name="pm", inside="cyt", outside="ec"),
        ),
    )


def _equation(variable: str, subdomain: str, ic: str, source: str, template: str = "bulk_radv_diff") -> str:
    terms = f'uniform_rate: "{source}"' if template == "region_ode" else f'diffusion: "0.1", source: "{source}"'
    return (
        f"    - template: {template}\n      variable: {variable}\n      subdomain: {subdomain}\n"
        f'      temporality: time_dependent\n      terms: {{ {terms} }}\n      initial_condition: "{ic}"\n'
    )


_SIZES = {"c": 2.0, "d": 3.0e-9, "e": 5.0e-4, "r": 7.0e-6, "N": 11.0}


def _decaying_cell(membrane_species: bool) -> str:
    """Every variable decays from a uniform start of its own size (`_SIZES`), so its running maximum is its start.
    With a membrane species the solver goes matrix-free and holds c and d as components of one cytosol block."""
    variables = [("N", "nuc", "region"), ("c", "cyt", None), ("d", "cyt", None), ("e", "ec", None)]
    equations = _equation("N", "nuc", "11.0", "-N", "region_ode")
    equations += _equation("c", "cyt", "2.0", "-c") + _equation("d", "cyt", "3e-9", "-2 * d")
    equations += _equation("e", "ec", "5e-4", "-0.5 * e")
    if membrane_species:
        variables.append(("r", "pm", None))
        equations += _equation("r", "pm", "7e-6", "-r", "surface_pde_with_dilution")
    lines = "".join(
        f"    - {{ name: {n}, subdomain: {s}" + (f", space: {sp}" if sp else "") + " }\n" for n, s, sp in variables
    )
    return (
        "math_description:\n  geometry: cell\n  subdomains:\n"
        + "".join(f"    - {{ name: {n}, kind: volume, motion: {{ kind: none }} }}\n" for n in ("nuc", "cyt", "ec"))
        + "    - { name: ne, kind: surface }\n    - { name: pm, kind: surface }\n"
        + f"  variables:\n{lines}  equations:\n{equations}"
    )


@pytest.mark.parametrize("membrane_species", [False, True], ids=["scalar blocks", "matrix-free components"])
def test_the_multi_compartment_solver_gives_each_variable_its_own_scale(membrane_species: bool) -> None:
    geometry = realize_multi_compartment(_nested_cell(), h=0.3)
    scales = SpeciesScales()
    integrate_multi_compartment(
        load_yaml(_decaying_cell(membrane_species)), geometry, t_final=0.2, tolerance_scales=scales
    )
    expected = {name: size for name, size in _SIZES.items() if membrane_species or name != "r"}
    assert scales.peak == pytest.approx(expected, rel=1.0e-12)


def test_the_single_mesh_integrator_gives_each_component_its_own_scale() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    model = (
        "math_description:\n  geometry: disk_2d\n  subdomains:\n"
        "    - { name: cyto, kind: volume, motion: { kind: none } }\n"
        "  variables:\n    - { name: c, subdomain: cyto }\n    - { name: d, subdomain: cyto }\n  equations:\n"
        + _equation("c", "cyto", "2.0", "-c")
        + _equation("d", "cyto", "3e-9", "-2 * d")
    )
    problem = assemble(load_yaml(model), geometry, dt=0.1, fe_degree=1)
    scales = SpeciesScales()
    integrate_discrete_problem(problem, t_final=0.2, tolerance_scales=scales)
    assert scales.peak == pytest.approx({"c": 2.0, "d": 3.0e-9}, rel=1.0e-12)


# --- a species twelve decades below another, in a coupled finite-element solve --------------------------------------


def _relative_l2(a: fem.Function, b: fem.Function) -> float:
    mesh = b.function_space.mesh
    difference = fem.Function(b.function_space)
    difference.x.array[:] = a.x.array - b.x.array

    def integral(f: UflExpr) -> float:
        return float(fem.assemble_scalar(fem.form(f * ufl.dx(domain=mesh))).real)

    return math.sqrt(integral(difference * difference) / integral(b * b))


def test_a_species_twelve_decades_down_is_controlled() -> None:
    """ADR 022's two-species table: an O(1) cytosol species beside an extracellular one of size 1e-12, diffusing
    from a non-uniform start. One scalar atol (1e-8) leaves the small one uncontrolled; its own holds it to rtol."""
    model = (
        "math_description:\n  geometry: cell\n  subdomains:\n"
        + "".join(f"    - {{ name: {n}, kind: volume, motion: {{ kind: none }} }}\n" for n in ("nuc", "cyt", "ec"))
        + "    - { name: ne, kind: surface }\n    - { name: pm, kind: surface }\n"
        + "  variables:\n    - { name: c, subdomain: cyt }\n    - { name: e, subdomain: ec }\n  equations:\n"
        + _equation("c", "cyt", "1.0", "-0.05 * c")
        + _equation("e", "ec", "1e-12 * (1.0 + 0.5 * cos(6 * geom.x[0]))", "0.0")
    )
    md = load_yaml(model)
    geometry = realize_multi_compartment(_nested_cell(), h=0.15)
    reference = integrate_multi_compartment(md, geometry, t_final=1.0, rtol=1.0e-10, atol=1.0e-22)
    errors = {}
    for scaling in _SCALINGS:
        result = integrate_multi_compartment(md, geometry, t_final=1.0, tolerance_scaling=scaling)
        errors[scaling] = _relative_l2(result.fields["e"], reference.fields["e"])
    assert errors["species"] < 1.0e-4, errors
    assert errors["scalar"] > 10.0 * errors["species"], errors


# --- slow: mesh independence and MPI ------------------------------------------------------------------------------


@pytest.mark.integration
def test_a_region_variable_is_controlled_alike_on_every_mesh() -> None:
    """ADR 022's nucleus model: a well-mixed nucleus ``N' = −3N`` drives the step beside two slowly decaying bulk
    species. Under one 2-norm over every unknown its share of the norm falls as the mesh is refined (measured
    5.4e-3, 7.0e-3, 9.4e-3); balanced, its error is the same on every mesh (5.9e-4)."""
    model = (
        "math_description:\n  geometry: cell\n  subdomains:\n"
        + "".join(f"    - {{ name: {n}, kind: volume, motion: {{ kind: none }} }}\n" for n in ("nuc", "cyt", "ec"))
        + "    - { name: ne, kind: surface }\n    - { name: pm, kind: surface }\n"
        + "  variables:\n    - { name: N, subdomain: nuc, space: region }\n"
        + "    - { name: c, subdomain: cyt }\n    - { name: e, subdomain: ec }\n  equations:\n"
        + _equation("N", "nuc", "1.0", "-3.0 * N", "region_ode")
        + _equation("c", "cyt", "1.0", "-0.1 * c")
        + _equation("e", "ec", "1.0", "-0.1 * e")
    )
    md = load_yaml(model)
    exact = math.exp(-6.0)
    errors: dict[ToleranceScaling, list[float]] = {"species": [], "scalar": []}
    for h in (0.12, 0.06):
        geometry = realize_multi_compartment(_nested_cell(), h=h)
        for scaling in errors:
            result = integrate_multi_compartment(md, geometry, t_final=2.0, tolerance_scaling=scaling)
            errors[scaling].append(abs(float(result.fields["N"].x.array[0].real) - exact) / exact)
    assert max(errors["species"]) < 1.2 * min(errors["species"]), errors
    assert errors["scalar"][1] > 3.0 * errors["species"][1], errors


_MPI_PROBE = r"""
import json, sys
sys.path.insert(0, TESTS)
from mpi4py import MPI
from test_species_tolerances import _decaying_cell, _nested_cell
from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment
from vcell_fenics.backend.tolerances import SpeciesScales
from vcell_fenics.formalism import load_yaml

out = {}
for membrane in (False, True):
    scales = SpeciesScales()
    geometry = realize_multi_compartment(_nested_cell(), h=0.3)
    result = integrate_multi_compartment(
        load_yaml(_decaying_cell(membrane)), geometry, t_final=0.2, tolerance_scales=scales
    )
    out[str(membrane)] = {"steps": result.steps, "peak": scales.peak}
if MPI.COMM_WORLD.rank == 0:
    print("PROBE " + json.dumps(out))
"""


def _run_mpi(n: int) -> dict[str, Any]:
    mpiexec = Path(sys.executable).parent / "mpiexec"
    probe = _MPI_PROBE.replace("TESTS", repr(str(Path(__file__).parent)))
    result = subprocess.run(
        [str(mpiexec), "-n", str(n), sys.executable, "-c", probe], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, f"n={n} failed:\n{result.stderr[-4000:]}"
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PROBE "))
    parsed: dict[str, Any] = json.loads(line.removeprefix("PROBE "))
    return parsed


@pytest.mark.integration
def test_the_scales_are_global_under_mpi() -> None:
    """Each species' scale is its maximum over every rank (the region variable's on its owning rank only), and the
    steps match the serial run's: the tolerances do not depend on the partition."""
    serial, parallel = _run_mpi(1), _run_mpi(2)
    for key in serial:
        assert parallel[key]["peak"] == pytest.approx(serial[key]["peak"], rel=1.0e-12), key
        assert abs(parallel[key]["steps"] - serial[key]["steps"]) <= max(2, serial[key]["steps"] // 10), key
