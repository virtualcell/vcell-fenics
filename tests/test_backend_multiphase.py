"""Verification of the two-phase overdamped solve with interphase drag (multiphase step 2).

`solve_two_phase_overdamped` solves two inertia-free velocity fields — network `v_n` and
solvent `v_s` — coupled by an interphase drag `ξ(v_a − v_b)`, each with a normal-slip BC,
as one block system. The checks:

1. **Block + drag assembly** — a manufactured two-field solution (`v_n = [1,0]`, `v_s = 0`,
   so `f_n = ξ v_n`, `f_s = −ξ v_n`) is recovered to round-off.
2. **Drag locks the phases** — equal-and-opposite forcing drives the phases apart, and the
   relative slip `|v_n − v_s|` shrinks as the drag `ξ` grows (toward a common velocity).
3. **The drag is the only coupling** — at `ξ = 0` each phase reduces to the independent
   single-phase slip solve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import (
    make_disk_geometry,
    solve_overdamped_slip,
    solve_two_phase_overdamped,
    solve_two_phase_stokes,
)


def _disk(h: float = 0.05):  # type: ignore[no-untyped-def]
    return make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")


def _kinetic(v: fem.Function) -> float:
    mesh = v.function_space.mesh
    local = fem.assemble_scalar(fem.form(ufl.inner(v, v) * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))


def test_recovers_a_manufactured_two_field_solution() -> None:
    # v_n = [1,0], v_s = [0,0] ⇒ f_n = ξ(v_n − v_s) = ξ[1,0], f_s = −ξ[1,0]; g_n = [1,0]·n,
    # g_s = 0. Recovering both verifies the block + drag + slip assembly.
    mesh = _disk()
    n = ufl.FacetNormal(mesh)
    e1 = ufl.as_vector([1.0, 0.0])
    xi = 5.0
    v_n, v_s = solve_two_phase_overdamped(
        mesh,
        drag=xi,
        forcing_n=xi * e1,
        forcing_s=-xi * e1,
        normal_velocity_n=ufl.dot(e1, n),
        normal_velocity_s=0.0 * ufl.SpatialCoordinate(mesh)[0],
    )
    assert np.abs(v_n.x.array.reshape(-1, 2) - np.array([1.0, 0.0])).max() < 1e-10
    assert np.abs(v_s.x.array).max() < 1e-10


def test_interphase_drag_locks_the_phases() -> None:
    # Equal-opposite rotational forcing (no-penetration on both, a small substrate friction
    # to fix the rotational null space): the slip |v_n − v_s| shrinks ~1/ξ as drag grows.
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    rotation = ufl.as_vector([-x[1], x[0]])
    g0 = 0.0 * x[0]

    def _slip(xi: float) -> float:
        v_n, v_s = solve_two_phase_overdamped(
            mesh,
            drag=xi,
            forcing_n=rotation,
            forcing_s=-rotation,
            normal_velocity_n=g0,
            normal_velocity_s=g0,
            screening_n=0.5,
            screening_s=0.5,
        )
        return float(np.abs(v_n.x.array - v_s.x.array).max())

    weak, medium, strong = _slip(0.5), _slip(5.0), _slip(50.0)
    assert strong < medium < weak  # more drag ⇒ tighter lock
    assert strong < 0.1 * weak  # an order of magnitude tighter from ξ=0.5 to 50


def test_zero_drag_decouples_into_single_phase_solves() -> None:
    # With ξ = 0 the off-diagonal coupling vanishes, so each phase is exactly the
    # single-phase slip solve. Compared by kinetic energy (the collapsed mixed-subspace and
    # standalone dof orderings differ, but the field — hence ∫|v|² — must match).
    mesh = _disk()
    x = ufl.SpatialCoordinate(mesh)
    rotation = ufl.as_vector([-x[1], x[0]])
    g0 = 0.0 * x[0]

    v_n, v_s = solve_two_phase_overdamped(
        mesh,
        drag=0.0,
        forcing_n=rotation,
        forcing_s=-rotation,
        normal_velocity_n=g0,
        normal_velocity_s=g0,
        screening_n=0.5,
        screening_s=0.5,
    )
    reference = solve_overdamped_slip(mesh, forcing=rotation, normal_velocity=g0, viscosity=1.0, screening=0.5)

    assert _kinetic(v_n) == pytest.approx(_kinetic(reference), rel=1e-9)
    assert np.abs(v_n.x.array + v_s.x.array).max() < 1e-9  # ±forcing ⇒ mirror-image phases


# ---------------------------------------------------------------------------
# Step 3c — the two-phase INCOMPRESSIBLE mixture (drag block + Stokes pressure).
# ---------------------------------------------------------------------------


def _l2_div(u: fem.Function) -> float:
    mesh = u.function_space.mesh
    local = fem.assemble_scalar(fem.form(ufl.div(u) ** 2 * ufl.dx(domain=mesh)))
    return math.sqrt(float(mesh.comm.allreduce(local, op=MPI.SUM)))


@pytest.mark.parametrize("symmetric", [True, False])
def test_two_phase_stokes_recovers_a_manufactured_mixture(symmetric: bool) -> None:
    # v_n = [1,0], v_s = [-1,0] ⇒ v_n + v_s = 0 (mixture divergence-free); p = x. The strong
    # network momentum -∇·(2νε) + ξ(v_n-v_s) + ∇p = f_n gives f_n = ξ[2,0] + [1,0]; likewise
    # f_s = ξ[-2,0] + [1,0]. Slip g_a = v_a·n. All three fields recovered to round-off.
    mesh = _disk(h=0.06)
    n = ufl.FacetNormal(mesh)
    xi = 5.0
    e_plus, e_minus = ufl.as_vector([1.0, 0.0]), ufl.as_vector([-1.0, 0.0])
    v_n, v_s, p = solve_two_phase_stokes(
        mesh,
        drag=xi,
        forcing_n=ufl.as_vector([2 * xi + 1.0, 0.0]),
        forcing_s=ufl.as_vector([-2 * xi + 1.0, 0.0]),
        normal_velocity_n=ufl.dot(e_plus, n),
        normal_velocity_s=ufl.dot(e_minus, n),
        symmetric=symmetric,
    )
    assert np.abs(v_n.x.array.reshape(-1, 2) - np.array([1.0, 0.0])).max() < 1e-9
    assert np.abs(v_s.x.array.reshape(-1, 2) - np.array([-1.0, 0.0])).max() < 1e-9

    p_coords = p.function_space.tabulate_dof_coordinates()[:, 0]
    p_error = (p.x.array - p.x.array.mean()) - (p_coords - p_coords.mean())
    assert np.abs(p_error).max() < 1e-8

    total = fem.Function(v_n.function_space)
    total.x.array[:] = v_n.x.array + v_s.x.array
    assert _l2_div(total) < 1e-9  # the mixture is incompressible


def test_two_phase_stokes_mixture_is_incompressible_on_a_well_posed_flow() -> None:
    # A generic null-mode-orthogonal forcing with substrate friction: the mixture velocity
    # v_n + v_s is divergence-free to round-off (the pressure does its job across both phases).
    mesh = _disk(h=0.04)
    x = ufl.SpatialCoordinate(mesh)
    v_n, v_s, _ = solve_two_phase_stokes(
        mesh,
        drag=2.0,
        forcing_n=ufl.as_vector([1.0, 0.0]),
        forcing_s=ufl.as_vector([0.0, 1.0]),
        normal_velocity_n=0.0 * x[0],
        normal_velocity_s=0.0 * x[0],
        screening_n=1.0,
        screening_s=1.0,
    )
    total = fem.Function(v_n.function_space)
    total.x.array[:] = v_n.x.array + v_s.x.array
    assert _l2_div(total) < 1e-9
