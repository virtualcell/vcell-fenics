"""A moving front stated on the membrane drives the volume it encloses (ADR 013).

The membrane carries the prescribed velocity, which may read its outward normal `geom.normal`; the
enclosed volume's material is at rest (`motion: none`), its mesh follows the membrane, and its species
are transported in the lab frame relative to that mesh motion. The checks:

- **the normal** is the outward unit normal of the region the membrane encloses (§1.8.4);
- **an expanding circle and sphere** with `V_n = c` (`velocity: c * geom.normal`) grow at exactly `c`,
  and a species at rest keeps its mass (the conservative ALE time term);
- **at rest in the lab frame**: a linear profile stays where it is while the membrane translates, so the
  harmonic-extended interior mesh motion does not move the material (decision 3);
- **a remesh** keeps the membrane driving the rebuilt volume (decision 4);
- **refusals** (decisions 4 and 6): a driven volume bounded also by the box or by a nucleus, and a
  volume with its own motion beside the moving membrane.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend.ale import ALEState
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.validator import FormalismValidationError

_R0 = 0.5


def _cell(dim: int, *, nucleus: bool = False) -> GeometryDescription:
    r = " + ".join(f"geom.x[{k}]**2" for k in range(dim))
    subvolumes = [
        SubVolume(name="cyto", type="analytic", expression=f"{r} < {_R0**2}"),
        SubVolume(name="ext", type="analytic", expression="1.0"),
    ]
    surfaces = [SurfaceClass(name="pm", inside="cyto", outside="ext")]
    if nucleus:
        subvolumes.insert(0, SubVolume(name="nuc", type="analytic", expression=f"{r} < {(_R0 / 2) ** 2}"))
        surfaces.append(SurfaceClass(name="ne", inside="nuc", outside="cyto"))
    return GeometryDescription(
        name="cell",
        dim=dim,
        extent=(2.0, 2.0, 2.0 if dim == 3 else 1.0),
        origin=(-1.0, -1.0, -1.0 if dim == 3 else 0.0),
        subvolumes=tuple(subvolumes),
        surfaces=tuple(surfaces),
    )


def _model(
    velocity: str,
    *,
    home: str = "cyto",
    initial: str = "1.0",
    diffusion: str = "1.0",
    home_motion: str = "{ kind: none }",
    nucleus: bool = False,
) -> str:
    extra = "    - { name: nuc, kind: volume }\n    - { name: ne, kind: surface }\n" if nucleus else ""
    return f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: cyto, kind: volume, motion: {home_motion if home == "cyto" else "{ kind: none }"} }}
    - {{ name: ext, kind: volume, motion: {home_motion if home == "ext" else "{ kind: none }"} }}
    - {{ name: pm, kind: surface, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
{extra}  variables:
    - {{ name: u, subdomain: {home} }}
  equations:
    - template: bulk_radv_diff
      variable: u
      subdomain: {home}
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "{initial}"
"""


def _problem(velocity: str, *, dim: int = 2, h: float = 0.06, dt: float = 0.01, **kwargs: object) -> DiscreteProblem:
    problem = assemble(load_yaml(_model(velocity, **kwargs)), realize(_cell(dim), h=h), dt=dt)  # type: ignore[arg-type]
    assert isinstance(problem, DiscreteProblem)
    return problem


def _measure(problem: DiscreteProblem, integrand: object = 1.0) -> float:
    form = ufl.as_ufl(integrand) * ufl.dx(domain=problem.V.mesh)
    return float(fem.assemble_scalar(fem.form(form)).real)


def _run(problem: DiscreteProblem, steps: int) -> None:
    for k in range(1, steps + 1):
        problem.set_time(k * float(problem.dt.value))
        problem.step()


def test_the_volume_is_driven_by_its_membrane() -> None:
    problem = _problem("0.2 * geom.normal")
    assert problem.driving_surface == "pm"
    assert problem.motion is not None


def test_the_normal_points_out_of_the_enclosed_region() -> None:
    from vcell_fenics.backend.boundary_normal import BoundaryNormal

    problem = _problem("0.2 * geom.normal")
    normal = BoundaryNormal(problem.V.mesh)
    values = normal.function.x.array.reshape((-1, 2))
    coordinates = normal.function.function_space.tabulate_dof_coordinates()[:, :2]
    on_boundary = np.linalg.norm(values, axis=1) > 0.5
    radial = coordinates[on_boundary] / np.linalg.norm(coordinates[on_boundary], axis=1)[:, None]
    assert on_boundary.sum() > 20
    assert np.allclose(np.linalg.norm(values[on_boundary], axis=1), 1.0)
    assert np.einsum("ij,ij->i", values[on_boundary], radial).min() > 0.99


@pytest.mark.parametrize("dim", [2, 3])
def test_an_expanding_front_grows_at_its_normal_speed_and_keeps_the_mass(dim: int) -> None:
    speed, steps, dt = 0.2, 25, 0.02
    problem = _problem(f"{speed} * geom.normal", dim=dim, h=0.06 if dim == 2 else 0.15, dt=dt)
    mass0 = _measure(problem, problem.unknown)
    size0 = _measure(problem)
    _run(problem, steps)

    radius = _R0 + speed * steps * dt
    exact = math.pi * radius**2 if dim == 2 else 4.0 / 3.0 * math.pi * radius**3
    # the polygonal (polyhedral) domain under-measures the disk (ball) by O(h²), the same before and after
    assert _measure(problem) / exact == pytest.approx(
        size0 / (math.pi * _R0**2 if dim == 2 else 4 / 3 * math.pi * _R0**3), rel=1e-2
    )
    # the material is at rest and the front sweeps it: no species crosses the front (zero total flux
    # relative to it), so the mass is exactly the initial one, and it is spread over the larger domain
    assert _measure(problem, problem.unknown) == pytest.approx(mass0, rel=1e-10)


def test_a_species_at_rest_stays_where_it_is_while_the_membrane_translates() -> None:
    # u = x at rest: under a translating membrane the interior mesh moves (by the harmonic extension), but u
    # at each current position must stay x there. Riding the mesh instead would leave each node with its
    # *initial* x, off by the displacement 0.1. A linear profile does not diffuse, and D = 0.01 keeps the
    # front's boundary layer (√(D t) ≈ 0.045) thin while keeping the Galerkin cell Péclet number near 2.
    problem = _problem("[0.5, 0.0]", initial="geom.x[0]", diffusion="0.01", h=0.05, dt=0.01)
    _run(problem, 20)
    x = problem.V.tabulate_dof_coordinates()[:, 0]
    y = problem.V.tabulate_dof_coordinates()[:, 1]
    # away from the front: the material at rest piles up against the trailing wall and is depleted behind the
    # leading one (layers of width ~D/V = 0.02, whose influence reaches inward); inside r < 0.2 it is at rest
    interior = (x - 0.1) ** 2 + y**2 < 0.2**2
    assert interior.sum() > 20
    assert np.abs(problem.unknown.x.array[interior] - x[interior]).max() < 1e-3


def test_a_remesh_keeps_the_membrane_driving_the_volume() -> None:
    problem = _problem("0.2 * geom.normal", h=0.08, dt=0.02)
    md = load_yaml(_model("0.2 * geom.normal"))
    state = ALEState(problem=problem, md=md)
    _run(state.problem, 5)
    mass = _measure(state.problem, state.problem.unknown)
    state.remesh(0.08)
    assert state.problem.driving_surface == "pm" and state.problem.motion is not None
    size = _measure(state.problem)
    _run(state.problem, 5)
    assert _measure(state.problem) > size * 1.05  # still growing: the rebuilt volume follows the membrane
    assert _measure(state.problem, state.problem.unknown) == pytest.approx(mass, rel=1e-6)


def test_a_driven_volume_also_bounded_by_the_box_is_refused() -> None:
    with pytest.raises(FormalismValidationError, match="bounded by that membrane alone"):
        assemble(load_yaml(_model("0.2 * geom.normal", home="ext")), realize(_cell(2), h=0.1), dt=0.01)


def test_a_driven_volume_with_a_nucleus_inside_is_refused() -> None:
    with pytest.raises(FormalismValidationError, match="bounded by that membrane alone"):
        assemble(load_yaml(_model("0.2 * geom.normal", nucleus=True)), realize(_cell(2, nucleus=True), h=0.1), dt=0.01)


def test_a_volume_with_its_own_motion_beside_a_moving_membrane_is_refused() -> None:
    model = _model("[0.2, 0.0]", home_motion='{ kind: prescribed, velocity: "[0.2, 0.0]" }')
    with pytest.raises(FormalismValidationError, match="decision 6"):
        assemble(load_yaml(model), realize(_cell(2), h=0.1), dt=0.01)


def _boundary_spacing(problem: DiscreteProblem) -> np.ndarray:
    from vcell_fenics.backend.discrete import _boundary_loop

    assert problem.motion is not None
    space = problem.motion._space
    loop = _boundary_loop(space)
    assert loop is not None
    points = space.tabulate_dof_coordinates()[loop, :2]
    return np.asarray(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))


def test_tangential_node_motion_keeps_the_boundary_spacing() -> None:
    # A front moving only normally, retracting fast on one side (V_n = -0.6 at x = 0.5, 0 at x = -0.5): without
    # tangential node motion the retracting side's nodes gather; with it (the default on a driven volume) each
    # boundary node keeps its fraction of the boundary's length, so the spacing pattern the mesher made is kept,
    # and the domain shrinks as with the nodes left where the normal velocity puts them (only the normal motion
    # is physical).
    velocity = "-0.3 * (1 + 2 * geom.x[0]) * geom.normal"
    spread, lost = {}, {}
    for redistribute in (True, False):
        problem = _problem(velocity, h=0.05, dt=0.01)
        assert problem.motion is not None
        if not redistribute:
            problem.motion._loop = None
        initial = _boundary_spacing(problem)
        size0 = _measure(problem)
        _run(problem, 40)
        spacing = _boundary_spacing(problem)
        spread[redistribute] = (spacing.max() / spacing.min()) / (initial.max() / initial.min())
        lost[redistribute] = size0 - _measure(problem)
    assert spread[True] == pytest.approx(1.0, abs=1e-3)  # the mesher's spacing pattern kept
    assert spread[False] > 1.5  # 4.7 / 2.8 measured
    # sliding along a spline through the nodes cuts no corners: 0.29921 vs 0.29929 measured (along the polyline
    # it was 0.305, 2 % more, and it grew with every step)
    assert lost[True] == pytest.approx(lost[False], rel=2e-3)
