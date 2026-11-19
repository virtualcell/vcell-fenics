"""A diffusion that loses its sign stops the run, with the cause named, on every route (ADR 021 decision 4).

The model: a species `c` on the unit disk with no flux out, starting at 1 and produced at a uniform rate 1, so
`c = 1 + t` exactly; its diffusion `1 − c/2` reads it and turns negative at `t = 1`, when `c` passes 2.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.backend.diffusion_check import (
    DiffusionCoefficient,
    DiffusionSignCheck,
    DiffusionSignError,
    raised_by,
)
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism import load_dict, load_yaml

_MODEL = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "DIFFUSION", source: "1" }
      initial_condition: "1.0"
"""


def _single_mesh(diffusion: str, t_final: float) -> int:
    md = load_yaml(_MODEL.replace("DIFFUSION", diffusion))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    problem = assemble(md, geometry, dt=0.1, fe_degree=1)
    return integrate_discrete_problem(problem, t_final=t_final).steps


def _time_in(message: str) -> float:
    found = re.search(r"t ≈ ([0-9.eE+-]+)", message)
    assert found is not None, message
    return float(found.group(1))


# --- the single mesh ---------------------------------------------------------------------------------------------


def test_a_diffusion_that_turns_negative_stops_a_single_mesh_run_naming_the_species() -> None:
    with pytest.raises(DiffusionSignError, match=r"the diffusion of 'c' became negative") as stopped:
        _single_mesh("1 - c/2", t_final=2.0)
    # stopped at the first accepted step past t = 1, where c = 1 + t passes 2
    assert 1.0 < _time_in(str(stopped.value)) < 1.5


def test_a_diffusion_that_stays_positive_runs_to_the_end() -> None:
    assert _single_mesh("1 - c/4", t_final=2.0) > 0  # c reaches 3 < 4


def test_a_diffusion_that_reads_no_species_is_not_checked() -> None:
    md = load_yaml(_MODEL.replace("DIFFUSION", "1 - geom.x[0]/2"))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    assert assemble(md, geometry, dt=0.1, fe_degree=1).diffusion_coefficients == ()


# --- the check itself --------------------------------------------------------------------------------------------


def _field(values: str) -> tuple[dmesh.Mesh, fem.Function]:
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    u = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))
    u.interpolate(lambda x: eval(values, {"x": x, "np": np}))  # the test's own literal
    return mesh, u


def test_the_least_value_is_found_at_the_vertices_over_every_rank() -> None:
    mesh, u = _field("1 - x[0]")  # 0 at x = 1, affine: exact at the vertices
    check = DiffusionSignCheck([DiffusionCoefficient("u", 0.5 - u, mesh)])
    [(name, least)] = check.least()
    assert name == "u" and least == pytest.approx(-0.5)


def test_a_tensor_diffusion_is_checked_by_its_least_eigenvalue() -> None:
    mesh, u = _field("x[0]")
    # eigenvalues 1 ± u: the least reaches 0 at x = 1, and −0.25 with the shift
    tensor = ufl.as_matrix([[1.0, u], [u, 1.0]]) - 0.25 * ufl.Identity(2)
    check = DiffusionSignCheck([DiffusionCoefficient("u", tensor, mesh)])
    [(name, least)] = check.least()
    assert name == "u" and least == pytest.approx(-0.25)


def test_a_coefficient_at_exactly_zero_is_let_through() -> None:
    mesh, u = _field("x[0]")
    check = DiffusionSignCheck([DiffusionCoefficient("u", u, mesh)])
    check.check(0.0)  # degenerate, not backwards


def test_the_failure_path_names_the_diffusion_on_the_last_iterate() -> None:
    """When Newton's iterate crosses zero before any step is accepted, the integrator fails first; its failure
    path asks the check about the iterate the Functions hold."""
    mesh, u = _field("x[0] - 0.75")
    check = DiffusionSignCheck([DiffusionCoefficient("m", 0.5 - u, mesh)])  # 0.5 - 0.25 > 0 everywhere
    ts = PETSc.TS().create(PETSc.COMM_WORLD)
    ts.setTime(0.3)
    assert check.explain(ts) is None
    u.x.array[:] += 1.0  # the last iterate
    message = check.explain(ts)
    assert message is not None and "'m'" in message and "t ≈ 0.3" in message
    ts.destroy()


def test_the_error_a_monitor_raised_is_found_in_the_petsc_error_chain() -> None:
    stopped = DiffusionSignError("the diffusion of 'm' became negative")
    try:
        try:
            raise stopped
        except DiffusionSignError as inner:
            raise PETSc.Error(101) from inner
    except PETSc.Error as outer:
        assert raised_by(outer) is stopped
    assert raised_by(RuntimeError("unrelated")) is None


# --- the multi-compartment solver --------------------------------------------------------------------------------

_TWO = """
math_description:
  geometry: disk_in_medium
  subdomains:
    - { name: ec, kind: volume }
    - { name: cell, kind: volume }
    - { name: pm, kind: surface }
  variables:
    - { name: c, subdomain: cell }
    - { name: e, subdomain: ec }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cell
      temporality: time_dependent
      terms: { diffusion: "1 - c/2", source: "1" }
      initial_condition: "1.0"
    - template: bulk_radv_diff
      variable: e
      subdomain: ec
      temporality: time_dependent
      terms: { diffusion: "1 - c_never/2" }
      initial_condition: "0"
"""


def _disk_in_medium() -> object:
    from vcell_fenics.backend.multi_compartment import realize_multi_compartment
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    return realize_multi_compartment(
        GeometryDescription(
            name="disk_in_medium",
            dim=2,
            extent=(3.0, 3.0, 1.0),
            origin=(-1.5, -1.5, 0.0),
            subvolumes=(
                SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
                SubVolume(name="ec", type="analytic", expression="1.0"),
            ),
            surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
        ),
        h=0.3,
    )


def test_a_diffusion_that_turns_negative_stops_a_multi_compartment_run_naming_the_species() -> None:
    """The cell's `c = 1 + t` crosses 2 at t = 1; the medium's species, whose diffusion reads no species, is
    not checked."""
    from vcell_fenics.backend.multi_compartment import integrate_multi_compartment

    md = load_yaml(_TWO.replace("1 - c_never/2", "1"))
    with pytest.raises(DiffusionSignError, match=r"the diffusion of 'c' became negative") as stopped:
        integrate_multi_compartment(md, _disk_in_medium(), t_final=2.0)  # type: ignore[arg-type]
    assert 1.0 < _time_in(str(stopped.value)) < 1.5


def test_a_diffusion_that_reads_a_region_variable_and_a_parameter_is_checked() -> None:
    """A region variable lives on the parent mesh and the species on the compartment's: the check reads it there.
    `R = t` and `D = k·(1 − c·R)` with `c = 1`, so D turns negative at t = 1."""
    from vcell_fenics.backend.multi_compartment import integrate_multi_compartment

    raw = yaml.safe_load(_TWO.replace("1 - c_never/2", "1"))
    md = raw["math_description"]
    md["variables"].append({"name": "R", "subdomain": "cell", "space": "region"})
    md["equations"][0]["terms"] = {"diffusion": "k * (1 - c * R)", "source": "0"}
    md["equations"].append(
        {
            "template": "region_ode",
            "variable": "R",
            "subdomain": "cell",
            "temporality": "time_dependent",
            "terms": {"uniform_rate": "1"},
            "initial_condition": "0",
        }
    )
    md["parameters"] = [{"name": "k", "value": 2.0}]
    with pytest.raises(DiffusionSignError, match=r"the diffusion of 'c' became negative") as stopped:
        integrate_multi_compartment(load_dict(raw), _disk_in_medium(), t_final=2.0)  # type: ignore[arg-type]
    assert 1.0 < _time_in(str(stopped.value)) < 1.5
