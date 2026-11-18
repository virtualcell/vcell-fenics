"""The method of lines honours `rtol` as relative and `atol` as absolute.

petsc4py's signature is `TS.setTolerances(rtol, atol)`. From 2026-06-16 until this test, every integrator passed
them positionally as `(atol, rtol)`, so PETSc ran with them swapped. A uniformly decaying species of size 1e-3,
`c' = −c`, tells the two apart: with `rtol = 1e-6`, `atol = 1e-12` the relative tolerance controls it; swapped,
the absolute 1e-6 is 1e-3 of its size and controls it instead. A uniform field is exact in P1 and does not
diffuse, so the time integration is the only error.

Measured 2026-11-18 on the single mesh: relative error 1.4e-4 with the tolerances as meant, 2.2e-2 swapped.
"""

from __future__ import annotations

import math

import numpy as np

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_SCALE, _T = 1.0e-3, 2.0
_EXACT = _SCALE * math.exp(-_T)

_SINGLE = f"""
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: none }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "1", source: "-c" }}
      initial_condition: "{_SCALE}"
"""

_TWO = f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: cyt, kind: volume, motion: {{ kind: none }} }}
    - {{ name: ec, kind: volume, motion: {{ kind: none }} }}
    - {{ name: pm, kind: surface }}
  variables:
    - {{ name: c, subdomain: cyt }}
    - {{ name: e, subdomain: ec }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyt
      temporality: time_dependent
      terms: {{ diffusion: "1", source: "-c" }}
      initial_condition: "{_SCALE}"
    - template: bulk_radv_diff
      variable: e
      subdomain: ec
      temporality: time_dependent
      terms: {{ diffusion: "1", source: "-e" }}
      initial_condition: "{_SCALE}"
"""


def _relative_error(values: np.ndarray) -> float:
    return float(np.abs(values - _EXACT).max() / _EXACT)


def test_the_single_mesh_integrator_takes_rtol_as_relative() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    problem = assemble(load_yaml(_SINGLE), geometry, dt=0.1, fe_degree=1)
    integrate_discrete_problem(problem, t_final=_T, rtol=1e-6, atol=1e-12)
    assert _relative_error(problem.unknown.x.array) < 1e-3


def test_the_multi_compartment_integrator_takes_rtol_as_relative() -> None:
    description = GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cyt", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.36"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyt", outside="ec"),),
    )
    geometry = realize_multi_compartment(description, h=0.3)
    result = integrate_multi_compartment(load_yaml(_TWO), geometry, t_final=_T, rtol=1e-6, atol=1e-12)
    for name in ("c", "e"):
        assert _relative_error(result.fields[name].x.array) < 1e-3, name
