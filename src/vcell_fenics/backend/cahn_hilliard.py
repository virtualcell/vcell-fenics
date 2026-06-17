"""Cahn–Hilliard phase separation — a prototype diffuse-interface template.

A first test of the "new physics enters as a template" maturation pipeline
(`docs/modeling/validation-and-diagnostics.md` §1.1): an expert-authored weak-form expansion,
which a later increment can *solidify* into a first-class formal template (a registry entry,
reserved names, mixed-space assembly through the formalism).

**The phases are partially coincident, not disjoint.** Unlike the sharp-interface ALE membrane
(two spatially *disjoint* regions across a zero-width boundary, `backend/fsi.py`) or the
two-phase cytoplasm mixture (two velocity fields *coincident everywhere* as volume fractions,
`backend/multiphase.py`), Cahn–Hilliard is **one conserved order parameter** `φ` whose two
phases — the wells of a double-well free energy, here `φ = 0` and `φ = 1` — meet at a **diffuse
interface** of width ~`ε`, where they mix. The relevant biology is liquid–liquid phase
separation (biomolecular condensates / membraneless organelles).

The PDE is 4th order,

    ∂φ/∂t = ∇·(M ∇μ),   μ = f'(φ) − ε² ∇²φ,   f(φ) = W φ²(1 − φ)²,

handled as a **mixed `(φ, μ)` system** — two coupled 2nd-order weak forms on a `[P1, P1]` space
(`φ`/`μ` same order; this mixed structure is *not* inf-sup-constrained, unlike Stokes), solved
with Newton each step (`f'` is nonlinear). A `θ`-scheme on the chemical potential gives the time
discretisation. Two properties make it verifiable: the order parameter is **exactly conserved**
(`∮` no-flux ⇒ `d/dt ∫φ = 0`), and the **free energy** `F = ∫[f(φ) + ε²/2 |∇φ|²] dx` is a
Lyapunov functional of the continuous dynamics (Cahn–Hilliard is its gradient flow) that the
discretisation **decreases for a sufficiently small step** — the fully-implicit treatment of
`f'` is not unconditionally energy-stable (convex splitting would be; out of scope for a
prototype).

Verified (`tests/test_backend_cahn_hilliard.py`): from a near-uniform spinodal initial state the
field separates toward the two wells across diffuse interfaces, `∫φ` is conserved to round-off,
and the free energy decreases monotonically.
"""

from __future__ import annotations

import basix.ufl
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import Mesh
from mpi4py import MPI

from vcell_fenics.backend._typing import UflExpr

# Height of the double-well `W φ²(1−φ)²`; its wells (the two phases) are at φ = 0 and φ = 1, and
# the spinodal (unstable) band where it phase-separates is φ ∈ ((3−√3)/6, (3+√3)/6) ≈ (0.21, 0.79).
_WELL_HEIGHT = 100.0


def _double_well_derivative(phi: UflExpr) -> UflExpr:
    """`f'(φ)` for `f(φ) = W φ²(1−φ)²` — i.e. `2W φ(1−φ)(1−2φ)`."""
    return 2.0 * _WELL_HEIGHT * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def cahn_hilliard_free_energy(phi: fem.Function, *, epsilon: float) -> float:
    """The Cahn–Hilliard free energy `F = ∫[W φ²(1−φ)² + ε²/2 |∇φ|²] dx` — the bulk double-well
    plus the gradient (interface) penalty. The Lyapunov functional the dynamics decrease."""

    mesh = phi.function_space.mesh
    bulk = _WELL_HEIGHT * phi**2 * (1.0 - phi) ** 2
    gradient = 0.5 * epsilon**2 * ufl.dot(ufl.grad(phi), ufl.grad(phi))
    local = fem.assemble_scalar(fem.form((bulk + gradient) * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))


def solve_cahn_hilliard(
    mesh: Mesh,
    *,
    initial: fem.Function,
    dt: float,
    n_steps: int,
    mobility: float = 1.0,
    epsilon: float = 0.1,
    theta: float = 1.0,
) -> tuple[fem.Function, list[float]]:
    """Integrate Cahn–Hilliard for `n_steps` steps of `dt`, returning the final order parameter
    `φ` and the free-energy history (one value per step, plus the initial).

    `initial` is a scalar P1 `Function` holding the initial `φ` (e.g. `0.5 + small noise`, in the
    spinodal band, for spinodal decomposition). `mobility` is `M`, `epsilon` is the interface
    width scale `ε` (so the gradient-energy coefficient is `ε²`), and `theta` weights the chemical
    potential in time (`1` = backward Euler, the default — most dissipative, so the free energy
    decreases for the widest range of steps; `0.5` = Crank–Nicolson is 2nd-order but more prone
    to a transient energy bump). Neither is *unconditionally* energy-stable with the fully-implicit
    `f'`, so the step must be small enough. No-flux (natural) boundary, so `φ` is conserved. Each
    step is a Newton solve of the coupled `(φ, μ)` residual.
    """

    p1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    mixed = fem.functionspace(mesh, basix.ufl.mixed_element([p1, p1]))
    solution = fem.Function(mixed)  # (φ, μ) at the new step
    previous = fem.Function(mixed)  # (φ, μ) at the old step
    phi, mu = ufl.split(solution)
    phi_old, mu_old = ufl.split(previous)
    test_phi, test_mu = ufl.TestFunctions(mixed)
    dx = ufl.Measure("dx", domain=mesh)

    solution.sub(0).interpolate(initial)  # φ from the IC; μ starts at 0
    solution.x.scatter_forward()
    previous.x.array[:] = solution.x.array

    mu_mid = (1.0 - theta) * mu_old + theta * mu
    # φ equation: ∂φ/∂t = ∇·(M∇μ)  ⇒  ∫(φ − φ⁰)q + dt·M ∫∇μ·∇q = 0
    residual = (phi - phi_old) * test_phi * dx + dt * mobility * ufl.dot(ufl.grad(mu_mid), ufl.grad(test_phi)) * dx
    # μ definition: μ = f'(φ) − ε²∇²φ  ⇒  ∫μ·v − ∫f'(φ)·v − ε² ∫∇φ·∇v = 0
    residual += mu * test_mu * dx - _double_well_derivative(phi) * test_mu * dx
    residual -= epsilon**2 * ufl.dot(ufl.grad(phi), ufl.grad(test_mu)) * dx

    problem = NonlinearProblem(
        residual,
        solution,
        petsc_options_prefix=f"vcellfenics_cahnhilliard_{id(solution):x}_",
        petsc_options={"snes_type": "newtonls", "snes_rtol": 1.0e-9, "ksp_type": "preonly", "pc_type": "lu"},
    )

    phi_final = solution.sub(0).collapse()
    energies = [cahn_hilliard_free_energy(phi_final, epsilon=epsilon)]
    for _ in range(n_steps):
        problem.solve()
        previous.x.array[:] = solution.x.array
        phi_final = solution.sub(0).collapse()
        energies.append(cahn_hilliard_free_energy(phi_final, epsilon=epsilon))
    return phi_final, energies
