"""Lab-frame (Eulerian) species on a moving mesh — the ``advection`` slot (VCell moving-boundary semantics).

VCell's moving-boundary solver is purely kinematic: a fixed lab grid, a moving front, and each species'
own lab-frame velocity c. There is no mesh velocity in that physics. ALE introduces one, w, as
bookkeeping: the backend transports a lab-frame species relative to the mesh, ``c − w``, so the
solution must not depend on w — only on the front (``w·n = v_b·n`` there) and on c. These tests pin
that down against exact answers:

- **w-independence** — a disk whose mesh *rotates* (w·n = 0: the domain never changes) with species at
  rest in the lab must reproduce the static-mesh solution. The Lagrangian default (species ride the
  mesh) would rotate the field instead, which is the negative control.
- **the travelling exponential** — species at rest in a disk translating at V: in the disk's frame they
  drift at −V and pile against the *trailing* wall; the steady profile is ``u ∝ exp(−V·ξ/D)``.
- **carrying** — a species whose lab velocity equals the (uniform) front velocity rides with the cell,
  identically to the Lagrangian default.
- **conservation** — the zero-total-flux (Rankine–Hugoniot) boundary keeps every species' mass exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem, geometry
from numpy.typing import NDArray

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.formalism import load_yaml

_DT = 0.01


def _problem(*, motion: str | None, terms: str, initial: str, h: float = 0.08) -> DiscreteProblem:
    motion_yaml = f'{{ kind: prescribed, velocity: "{motion}" }}' if motion is not None else "{ kind: none }"
    model = f"""
math_description:
  geometry: disk
  subdomains:
    - {{ name: cyto, kind: volume, motion: {motion_yaml} }}
  variables:
    - {{ name: u, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: u
      subdomain: cyto
      temporality: time_dependent
      terms: {{ {terms} }}
      initial_condition: "{initial}"
"""
    problem = assemble(load_yaml(model), make_disk_geometry("disk", volume_subdomain="cyto", h=h), dt=_DT)
    assert isinstance(problem, DiscreteProblem)
    return problem


def _advance(problem: DiscreteProblem, t_final: float) -> None:
    for k in range(1, round(t_final / _DT) + 1):
        problem.set_time(k * _DT)
        problem.step()


def _mass(problem: DiscreteProblem) -> float:
    return float(fem.assemble_scalar(fem.form(problem.unknown * ufl.dx(domain=problem.V.mesh))).real)


def _eval(u: fem.Function, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """``u`` at lab points (N, 3) — each point must lie in u's mesh."""

    mesh = u.function_space.mesh
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    colliding = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, points), points)
    cells = np.array([colliding.links(np.int32(i))[0] for i in range(points.shape[0])], dtype=np.int32)
    values: NDArray[np.float64] = u.eval(points, cells).reshape(-1)
    return values


_BLOB = "exp(-((geom.x[0] - 0.4)**2 + geom.x[1]**2) / 0.05)"
_SPIN = "[-2.0 * geom.x[1], 2.0 * geom.x[0]]"  # rigid rotation at 2 rad per unit time: tangential on the circle


def _rotation_errors(h: float, dt: float) -> tuple[float, float]:
    """Max relative difference from the static-mesh solution at t = 0.5 (the mesh turned by 1 rad), for
    a lab-frame species on the rotating mesh and — the negative control — a species carried by it."""

    global _DT
    saved, _DT = _DT, dt
    try:
        terms_lab, terms_carried = 'diffusion: "0.05", advection: "[0.0, 0.0]"', 'diffusion: "0.05"'
        static = _problem(motion=None, terms=terms_lab, initial=_BLOB, h=h)
        rotating = _problem(motion=_SPIN, terms=terms_lab, initial=_BLOB, h=h)
        carried = _problem(motion=_SPIN, terms=terms_carried, initial=_BLOB, h=h)
        for problem in (static, rotating, carried):
            _advance(problem, 0.5)
    finally:
        _DT = saved
    # compare at the rotating mesh's current nodes, pulled inside both domains: each is a polygon inscribed
    # in the circle, and the rotating one has grown by its explicit node update (x += dt·v gains a factor
    # √(1 + (ω dt)²) per step — ~1 % at dt = 0.01, first order in dt)
    nodes: NDArray[np.float64] = np.array(rotating.V.mesh.geometry.x, dtype=np.float64, copy=True) * 0.97
    reference = _eval(static.unknown, nodes)
    scale = np.abs(reference).max()
    lab = float(np.abs(_eval(rotating.unknown, nodes) - reference).max() / scale)
    moved = float(np.abs(_eval(carried.unknown, nodes) - reference).max() / scale)
    return lab, moved


def test_the_solution_does_not_depend_on_the_mesh_velocity() -> None:
    # the lab-frame solution on a rotating mesh converges (first order: backward Euler) to the static-mesh
    # one, while a carried species stays rotated — O(1) away at every resolution
    coarse, carried_coarse = _rotation_errors(0.08, 0.01)
    fine, carried_fine = _rotation_errors(0.04, 0.005)
    assert fine < 0.025 and coarse / fine > 1.7, (coarse, fine)
    assert min(carried_coarse, carried_fine) > 0.5, (carried_coarse, carried_fine)


def test_species_at_rest_form_the_travelling_exponential_against_the_trailing_wall() -> None:
    # D = V: the steady profile in the disk's frame is u ∝ exp(−(x − x_c)), whatever the initial state
    problem = _problem(motion="[0.5, 0.0]", terms='diffusion: "0.5", advection: "[0.0, 0.0]"', initial="1.0", h=0.1)
    mass0 = _mass(problem)
    _advance(problem, 6.0)  # three diffusion times R²/D
    points = np.asarray(problem.V.tabulate_dof_coordinates())
    values = np.asarray(problem.unknown.x.array)
    xi = points[:, 0] - points[:, 0].mean()
    slope = np.polyfit(xi, np.log(values), 1)[0]
    assert slope == pytest.approx(-0.5 / 0.5, rel=0.05), slope  # −V/D: higher at the trailing (left) wall
    assert _mass(problem) == pytest.approx(mass0, rel=1e-12)


def test_a_species_moving_with_the_front_rides_with_the_cell() -> None:
    initial = "geom.x[0] + 2.0 * geom.x[1]"
    lab = _problem(motion="[0.5, 0.25]", terms='diffusion: "0.1", advection: "[0.5, 0.25]"', initial=initial)
    carried = _problem(motion="[0.5, 0.25]", terms='diffusion: "0.1"', initial=initial)
    for problem in (lab, carried):
        _advance(problem, 0.5)
    assert np.allclose(lab.unknown.x.array, carried.unknown.x.array, atol=1e-10)


def test_a_swept_species_keeps_its_mass() -> None:
    # a strongly deforming front and a species at rest: the zero-total-flux front keeps the mass exactly
    problem = _problem(
        motion="[0.3 * geom.x[0] * geom.x[0], 0.2 * geom.x[1]]",
        terms='diffusion: "0.2", advection: "[0.0, 0.0]"',
        initial=_BLOB,
    )
    mass0 = _mass(problem)
    _advance(problem, 0.5)
    assert _mass(problem) == pytest.approx(mass0, rel=1e-12)
