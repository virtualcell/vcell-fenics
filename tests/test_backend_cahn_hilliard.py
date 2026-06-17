"""Verification of the Cahn–Hilliard prototype (backend/cahn_hilliard) — diffuse-interface
phase separation, a first test of the "new physics as a template" pipeline.

The 4th-order equation is solved in its **mixed `(φ, μ)`** form — two coupled 2nd-order weak
forms. Three checks, sharing one integration (a module fixture):

1. **Conservation** — `φ` is a conserved order parameter (`∂φ/∂t = ∇·(M∇μ)`, no-flux), so `∫φ`
   is invariant to round-off, however far the field separates.
2. **Free energy relaxes** — the free energy `F = ∫[Wφ²(1−φ)² + ε²/2|∇φ|²]` (the Lyapunov
   functional of the gradient flow) decreases monotonically at a small enough step.
3. **Phase separation** — a near-uniform state in the spinodal band separates toward the two
   wells (`φ ≈ 0` and `φ ≈ 1`) across diffuse interfaces.
"""

from __future__ import annotations

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import cahn_hilliard_free_energy, solve_cahn_hilliard


def _total(phi: fem.Function) -> float:
    mesh = phi.function_space.mesh
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(phi * ufl.dx(domain=mesh))), op=MPI.SUM))


@pytest.fixture(scope="module")
def separation() -> tuple[float, float, fem.Function, list[float]]:
    """One Cahn–Hilliard run from a near-uniform spinodal state; returns the initial `∫φ`, the
    initial peak-to-peak spread, the final `φ`, and the free-energy history."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 48, 48)
    initial = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))
    initial.x.array[:] = 0.5 + 0.05 * np.random.default_rng(42).standard_normal(initial.x.array.size)
    total0 = _total(initial)
    spread0 = float(initial.x.array.max() - initial.x.array.min())
    phi, energies = solve_cahn_hilliard(mesh, initial=initial, dt=2e-4, n_steps=80, epsilon=0.05)
    return total0, spread0, phi, energies


def test_cahn_hilliard_conserves_the_order_parameter(separation) -> None:  # type: ignore[no-untyped-def]
    total0, _, phi, _ = separation
    assert abs(_total(phi) - total0) < 1e-12  # ∫φ invariant to round-off — the conservative form


def test_cahn_hilliard_free_energy_decreases_monotonically(separation) -> None:  # type: ignore[no-untyped-def]
    *_, energies = separation
    assert all(energies[i + 1] <= energies[i] + 1e-12 for i in range(len(energies) - 1))  # Lyapunov
    assert energies[-1] < energies[0]  # and it actually relaxes


def test_cahn_hilliard_separates_into_two_phases(separation) -> None:  # type: ignore[no-untyped-def]
    _, spread0, phi, _ = separation
    assert spread0 < 0.5  # started near-uniform (in the spinodal band)
    assert float(phi.x.array.min()) < 0.2  # a phase near the φ = 0 well
    assert float(phi.x.array.max()) > 0.8  # a phase near the φ = 1 well — partially coincident, diffuse


def test_free_energy_helper_is_zero_at_a_well() -> None:
    # A uniform field at a well (φ = 0) has zero free energy (no bulk penalty, no gradient).
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    phi = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))  # φ = 0 everywhere
    assert abs(cahn_hilliard_free_energy(phi, epsilon=0.1)) < 1e-14
