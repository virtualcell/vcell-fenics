"""SolverConfiguration and the time-stepping driver.

A run is a triple `(MathDescription, Geometry, SolverConfiguration)` (doc §3.4):
the MathDescription is *what* is solved, the Geometry *where*, and the
SolverConfiguration *how approximately*. `SolverConfiguration` here carries the
v1-relevant discretisation choices — time step, final time, FE degree; the
scheme (backward Euler) and linear solver (direct LU) are backend defaults
(§3.3). The full §3.4 object (linear/nonlinear solver, ALE, stabilisation
knobs) and a YAML carrier for it are future work.

`run` assembles the problem and advances it to `t_final`, returning the solved
`DiscreteProblem` whose `unknown` holds the final state. Capturing snapshots at
intermediate output times (§3.2.4) is a future enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass

from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.formalism.schema import MathDescription


@dataclass(frozen=True)
class SolverConfiguration:
    """Discretisation choices for a run (the v1 subset of doc §3.4)."""

    dt: float
    t_final: float
    fe_degree: int = 1


def run(md: MathDescription, geometry: Geometry, config: SolverConfiguration) -> DiscreteProblem:
    """Assemble `md` against `geometry` and advance it to `config.t_final` with
    backward-Euler steps of `config.dt`. Returns the solved DiscreteProblem."""

    problem = assemble(md, geometry, dt=config.dt, fe_degree=config.fe_degree)
    for _ in range(round(config.t_final / config.dt)):
        problem.step()
    return problem
