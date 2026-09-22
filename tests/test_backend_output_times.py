"""Output-time hooks on the method-of-lines integrators (`backend/output_times.py`).

VCell asks for a result at every output time, while the method-of-lines integrators step adaptively.
The hook records the solution at each requested time from a TS monitor — interpolated with the
integrator's own dense output, or copied when a step lands on the time — so these tests pin that:

- every requested time is recorded, once, in order, and the recorded values match the analytic
  solution to the integrator's accuracy;
- recording does not perturb the integration (same step count and final state as an unmonitored run);
- the interface-coupled integrator conserves mass at every recorded time, not just at the end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend import assemble, integrate_interface_coupled, make_two_bulk_membrane_geometry
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism.geometry_io import load_geometry_yaml
from vcell_fenics.formalism.loader import load_dict
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)

_MODELS = Path(__file__).resolve().parent.parent / "examples" / "models"
_K, _KD = 0.5, 0.2  # the example model's k_conv and k_decay


def _decay_problem() -> DiscreteProblem:
    """The CLI example (u → v, v decays) with a *uniform* initial condition and no-flux walls, so the
    diffusion never acts and each species follows its ODE exactly: u = e^{-k t},
    v = k/(k_d − k)·(e^{-k t} − e^{-k_d t})."""

    document = yaml.safe_load((_MODELS / "diffusion2d_math.yaml").read_text())
    document["math_description"]["equations"][0]["initial_condition"] = "1.0"
    math_description = load_dict(document)
    geometry = realize(load_geometry_yaml(_MODELS / "diffusion2d_geom.yaml"), h=0.5)
    problem = assemble(math_description, geometry, dt=0.1)
    assert isinstance(problem, DiscreteProblem)
    return problem


def _exact(t: float) -> tuple[float, float]:
    u = math.exp(-_K * t)
    return u, _K / (_KD - _K) * (math.exp(-_K * t) - math.exp(-_KD * t))


def test_every_output_time_is_recorded_once_and_matches_the_ode() -> None:
    problem = _decay_problem()
    times = [0.25 * k for k in range(1, 9)]  # 0.25 … 2.0, the last one == t_final
    recorded: list[tuple[float, np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]] = []
    progress: list[float] = []

    def on_output(t: float, snapshot: fem.Function) -> None:  # u and v are components of one unknown
        recorded.append((t, snapshot.sub(0).collapse().x.array.copy(), snapshot.sub(1).collapse().x.array.copy()))

    integrate_discrete_problem(
        problem, t_final=2.0, output_times=times, on_output=on_output, on_progress=progress.append, rtol=1e-8
    )

    assert [t for t, _, _ in recorded] == times
    assert progress == sorted(progress) and progress[-1] == pytest.approx(2.0)
    for t, u, v in recorded:
        u_exact, v_exact = _exact(t)
        assert np.allclose(u, u_exact, rtol=1e-5), t
        assert np.allclose(v, v_exact, rtol=1e-4, atol=1e-7), t


def test_recording_does_not_perturb_the_integration() -> None:
    times = [0.1 * k for k in range(1, 21)]
    monitored, plain = _decay_problem(), _decay_problem()
    result_monitored = integrate_discrete_problem(
        monitored, t_final=2.0, output_times=times, on_output=lambda t, snap: None
    )
    result_plain = integrate_discrete_problem(plain, t_final=2.0)
    assert result_monitored.steps == result_plain.steps
    assert np.array_equal(monitored.unknown.x.array, plain.unknown.x.array)


def test_the_initial_state_is_recorded_when_asked_for() -> None:
    """TS calls its monitors once before the first step, so an output time of t_start records the
    initial condition — the interface-coupled integrator builds its IC internally, so its caller has
    no other way to write the t = 0 row."""

    recorded: list[float] = []
    integrate_discrete_problem(
        _decay_problem(), t_final=1.0, output_times=[0.0, 0.5, 1.0], on_output=lambda t, s: recorded.append(t)
    )
    assert recorded == [0.0, 0.5, 1.0]

    initial: list[float] = []

    def first_u(t: float, snapshot: fem.Function) -> None:
        if t == 0.0:
            initial.extend(snapshot.sub(0).collapse().x.array)

    integrate_discrete_problem(_decay_problem(), t_final=1.0, output_times=[0.0], on_output=first_u)
    assert np.allclose(initial, 1.0)  # the uniform initial condition, untouched by any step


def test_output_times_outside_the_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="beyond t_final"):
        integrate_discrete_problem(_decay_problem(), t_final=1.0, output_times=[0.5, 1.5], on_output=lambda t, s: None)


def _permeability_model() -> MathDescription:
    return MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="u_in", subdomain="cyto"), Variable(name="u_out", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=2.0)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u_in", boundary="membrane", expression="P * (u_out - u_in)"),
            BCInterfaceFlux(variable="u_out", boundary="membrane", expression="P * (u_in - u_out)"),
        ],
    )


def _mass(field: fem.Function) -> float:
    mesh = field.function_space.mesh
    return float(fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh))).real)


def test_interface_coupled_records_every_time_and_conserves_mass_at_each() -> None:
    geometry = make_two_bulk_membrane_geometry(
        "cell",
        inner="cyto",
        outer_subdomain="ext",
        membrane="mem",
        interface="membrane",
        outer="wall",
        inner_radius=0.5,
        outer_radius=1.0,
        h=0.12,
    )
    times = [0.2 * k for k in range(0, 6)]  # including t = 0: the IC is built inside the integrator
    masses: list[tuple[float, float, float]] = []

    def on_output(t: float, inner: fem.Function, outer: fem.Function) -> None:
        masses.append((t, _mass(inner), _mass(outer)))

    result = integrate_interface_coupled(
        _permeability_model(), geometry, t_final=1.0, output_times=times, on_output=on_output
    )

    assert [t for t, _, _ in masses] == pytest.approx(times)
    initial = math.pi * 0.5**2  # u_in = 1 on the inner disk, u_out = 0 outside
    for t, m_in, m_out in masses:
        assert m_in + m_out == pytest.approx(initial, rel=2e-2), t  # the faceted disk's area, within h
    totals = [m_in + m_out for _, m_in, m_out in masses]
    assert max(totals) - min(totals) < 1e-6 * initial  # conserved across the run, not just at the end
    assert [m_in for _, m_in, _ in masses] == sorted([m_in for _, m_in, _ in masses], reverse=True)  # draining
    assert result.time == pytest.approx(1.0)
