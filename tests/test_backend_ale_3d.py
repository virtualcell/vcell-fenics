"""3D moving meshes (the ALE path on tetrahedra): prescribed boundary motion, the harmonic interior
extension, the conservative time term and lab-frame transport — the same machinery as 2D, which had
only ever run on triangles."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_DT = 0.05


def _sphere_problem(velocity: str, terms: str, initial: str) -> DiscreteProblem:
    geometry = GeometryDescription(
        name="ball",
        dim=3,
        extent=(4.0, 4.0, 4.0),
        origin=(-2.0, -2.0, -2.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < 1.0"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )
    model = f"""
math_description:
  geometry: ball
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
    - {{ name: ext, kind: volume }}
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
    problem = assemble(load_yaml(model), realize(geometry, h=0.25), dt=_DT)
    assert isinstance(problem, DiscreteProblem)
    return problem


def _integral(problem: DiscreteProblem, integrand: object) -> float:
    return float(fem.assemble_scalar(fem.form(integrand * ufl.dx(domain=problem.V.mesh))).real)


def _advance(problem: DiscreteProblem, t_final: float) -> None:
    for k in range(1, round(t_final / _DT) + 1):
        problem.set_time(k * _DT)
        problem.step()


def test_a_translating_sphere_keeps_its_volume_and_moves_exactly() -> None:
    problem = _sphere_problem("[0.4, -0.2, 0.3]", 'diffusion: "0.1", advection: "[0.0, 0.0, 0.0]"', "1.0 + geom.x[0]")
    x = ufl.SpatialCoordinate(problem.V.mesh)
    volume0, mass0 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    centroid0 = np.array([_integral(problem, x[i]) for i in range(3)]) / volume0
    _advance(problem, 1.0)
    volume1, mass1 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    centroid1 = np.array([_integral(problem, x[i]) for i in range(3)]) / volume1
    assert volume1 == pytest.approx(volume0, rel=1e-12)  # a rigid motion: the tetrahedra are only moved
    assert np.allclose(centroid1 - centroid0, [0.4, -0.2, 0.3], atol=1e-12)
    assert mass1 == pytest.approx(mass0, rel=1e-12)  # lab-frame species swept by the front: the total is kept


def test_a_deforming_sphere_keeps_its_mass() -> None:
    # an axisymmetric squeeze at the equator (the 3D furrow's shape of motion)
    ring = "-0.5 * exp(-geom.x[1]**2 / 0.1)"
    velocity = f"[{ring} * geom.x[0], 0.0, {ring} * geom.x[2]]"
    problem = _sphere_problem(velocity, 'diffusion: "0.1", advection: "[0.0, 0.0, 0.0]"', "1.0 + geom.x[1]")
    volume0, mass0 = _integral(problem, 1.0), _integral(problem, problem.unknown)
    _advance(problem, 0.5)
    assert _integral(problem, problem.unknown) == pytest.approx(mass0, rel=1e-12)
    assert _integral(problem, 1.0) < volume0  # the equator moved inward
