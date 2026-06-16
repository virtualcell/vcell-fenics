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
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism.schema import MathDescription


@dataclass(frozen=True)
class SolverConfiguration:
    """Discretisation choices for a run (the v1 subset of doc §3.4).

    `time_integration` selects the integrator: `"backward_euler"` (the default — fixed-step,
    first-order implicit, `dt` is the step) or `"method_of_lines"` (PETSc `TS` adaptive BDF,
    `backend/reaction_diffusion.py`; `dt` seeds the adaptive controller and `t_final` is the
    target). Method-of-lines integrates *nonlinear* reactions directly (the inner Newton) and
    adapts the step to stiffness, at the cost of being fixed-domain only.
    """

    dt: float
    t_final: float
    fe_degree: int = 1
    time_integration: str = "backward_euler"


def run(md: MathDescription, geometry: Geometry, config: SolverConfiguration) -> DiscreteProblem:
    """Assemble `md` against `geometry` and advance it to `config.t_final`, returning the solved
    DiscreteProblem whose `unknown` holds the final state. `config.time_integration` picks the
    integrator: fixed-step `"backward_euler"` or adaptive `"method_of_lines"` (PETSc `TS`)."""

    problem = assemble(md, geometry, dt=config.dt, fe_degree=config.fe_degree)
    if config.time_integration == "method_of_lines":
        integrate_discrete_problem(problem, t_final=config.t_final, dt_initial=config.dt)
    elif config.time_integration == "backward_euler":
        for n in range(round(config.t_final / config.dt)):
            problem.set_time((n + 1) * config.dt)  # advance g(t) etc. to the step's time, then solve
            problem.step()
    else:
        raise ValueError(
            f"time_integration must be 'backward_euler' or 'method_of_lines', not {config.time_integration!r}"
        )
    return problem
