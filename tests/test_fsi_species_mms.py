"""Method-of-manufactured-solutions convergence check for the FSI **co-moving volume species**
transport (`backend/fsi._transport_co_moving_species`, the operator inside
`step_two_phase_fsi_with_species`).

The existing FSI species tests (`test_backend_fsi.py`) are strong on **conservation**, **equilibria**,
and **discrimination** — but conservation is *blind to a wrong-but-conservative operator* (exactly how
the interface-flux over-count survived for the whole history of the coupled BE solver, since equilibration
never pinned an absolute flux). This test pins the transport to a known analytical solution and measures
its **spatial convergence order**, so a sign error / missing term / mis-scaled dilution in the real code
fails the build even though `∫c` might still be conserved.

Design (isolates the transport from the Stokes solve so the velocity is exactly known):
- **Manufactured solution** — `u*(x, t) = decay(t)·(2 + sin(x)·cos(y))`, either steady (`decay ≡ 1`) or
  **decaying** (`decay = e^{-t}`).
- **Carrier velocity** `v = [0.3·x, 0.1·y]` (∇·v = 0.4) **≠ frame velocity** `w = [0.25·x, 0.25·y]`, so *both*
  the relative advection `(v − w)·∇c` **and** the bulk dilution `c ∇·v` (the volume analogue of the
  mandatory surface `ρ ∇_Γ·v_Γ`) are active — every transport term is exercised. This carrier≠frame
  decoupling (an Eulerian species riding a *physical phase* while the mesh moves at the *bookkeeping*
  frame, diluting by ∇·v_carrier not ∇·v_mesh) is the FSI-distinctive structure — and the reason this is a
  pytest MMS rather than an `mms/` YAML case: the math-description formalism hard-wires dilution to the mesh
  velocity's GCL swept-volume (`assemble.py`), and `fsi.py` implements the transport inline, so a YAML case
  would exercise the formalism assembler (already covered by the `bulk_expansion` cases), not this operator.
- Both velocity fields are linear, so the ALE mesh's *harmonic* extension reproduces the frame exactly and
  `w` is closed-form; it **cancels analytically** in the forcing (transport is by the carrier):
  `f = ∂c*/∂t + ∇·(c*·v) − D∇²c*`. Time-dependent Dirichlet `u*` at the moving boundary; `D = 0.1`;
  `dt ∝ h²` so the O(dt) strided-motion floor stays below the spatial error.

Geometries follow the suite's box/disk convention: the straight **structured box** gives a clean O(h²), the
curvature-preserving **nested disk** the P1 curved-domain rate ~O(h^1.5). A fixed-domain Stokes-order check
(step 1) and the full force-balance-closure MMS remain follow-ups.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import CellType, Mesh, create_rectangle, exterior_facet_indices
from mpi4py import MPI

from vcell_fenics.backend.fsi import _harmonic_displacement, _transport_co_moving_species
from vcell_fenics.backend.geometry import make_nested_disk_geometry

_D = 0.1
_T_FINAL = 1.0
_RESOLUTIONS_H = [0.1, 0.05, 0.025]


def _u0(x: np.ndarray) -> np.ndarray:
    """The space part of the manufactured solution: 2 + sin(x)·cos(y)."""
    return np.asarray(2.0 + np.sin(x[0]) * np.cos(x[1]), dtype=float)


def _mesh(geometry: str, h: float) -> Mesh:
    """A straight-boundary structured box (clean O(h²)) or a nestable curved disk (~O(h^1.5))."""
    if geometry == "box":
        nx = round(2.0 / h)
        return create_rectangle(
            MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [nx, nx], CellType.triangle
        )
    return make_nested_disk_geometry("g", volume_subdomain="cyto", radius=1.0, h=h, base_h=0.1).mesh_of("cyto")


def _linf_error_at(
    h: float, *, geometry: str = "box", decaying: bool = False, dilution_in_source: bool = True
) -> float:
    """Advance the FSI co-moving species to t_final, return the nodal L∞ error vs the manufactured u*.

    `dilution_in_source=False` drops the dilution's contribution `c*·∇·v` from the manufactured forcing — a
    **negative control**: since the real operator still carries `c ∇·v`, u* is then no longer the exact
    solution and the error must stop converging, proving the positive test pins that cancellation-prone term."""
    mesh = _mesh(geometry, h)
    space = fem.functionspace(mesh, ("Lagrange", 1))
    c = fem.Function(space)
    c.interpolate(_u0)  # initial condition = c*(x, 0) = u0

    mesh.topology.create_connectivity(1, 2)
    boundary_dofs = fem.locate_dofs_topological(space, 1, exterior_facet_indices(mesh.topology))
    boundary_value = fem.Function(space)
    bc = fem.dirichletbc(boundary_value, boundary_dofs)

    x = ufl.SpatialCoordinate(mesh)  # symbolic — re-evaluated on the moved mesh each assemble
    carrier = ufl.as_vector([0.3 * x[0], 0.1 * x[1]])  # ∇·v = 0.4, ≠ frame
    frame = ufl.as_vector([0.25 * x[0], 0.25 * x[1]])  # → mesh velocity w (linear ⇒ harmonic ext exact)
    u0_ufl = 2.0 + ufl.sin(x[0]) * ufl.cos(x[1])
    steady_source = (
        0.3 * x[0] * ufl.cos(x[0]) * ufl.cos(x[1])
        - 0.1 * x[1] * ufl.sin(x[0]) * ufl.sin(x[1])
        + 0.8
        + 0.6 * ufl.sin(x[0]) * ufl.cos(x[1])
    )  # ∇·(u0·v) − D∇²u0, with D = 0.1
    time = fem.Constant(mesh, 0.0)
    decay = ufl.exp(-time) if decaying else fem.Constant(mesh, 1.0)
    # c* = decay·u0 ⇒ f = ∂c*/∂t + ∇·(c*·v) − D∇²c* = decay·(steady_source − u0·[decaying]).
    source = decay * (steady_source - (u0_ufl if decaying else fem.Constant(mesh, 0.0)))
    if not dilution_in_source:
        source = source - decay * (0.8 + 0.4 * ufl.sin(x[0]) * ufl.cos(x[1]))  # drop c*·∇·v (negative control)

    dt = 0.5 * h * h  # first-order in dt (BE + strided motion) ⇒ refine dt ∝ h² to expose the spatial order
    t_now = [0.0]

    def boundary_expr(pts: np.ndarray) -> np.ndarray:
        return (np.exp(-t_now[0]) if decaying else 1.0) * _u0(pts)

    for step in range(round(_T_FINAL / dt)):
        t_now[0] = (step + 1) * dt  # backward Euler: operator + BC at the new time
        time.value = t_now[0]
        boundary_value.interpolate(boundary_expr)  # refresh Dirichlet u* at the current (moved) boundary
        displacement = _harmonic_displacement(mesh, frame, dt)
        _transport_co_moving_species(
            mesh, c, carrier_velocity=carrier, displacement=displacement, dt=dt, diffusivity=_D, source=source, bcs=[bc]
        )

    coords = space.tabulate_dof_coordinates()  # final (moved) node positions, shape (npoints, 3)
    final = (np.exp(-_T_FINAL) if decaying else 1.0) * _u0(coords.T)
    return float(np.abs(c.x.array - final).max())


def _order(errors: list[float], resolutions: list[float]) -> float:
    return math.log(errors[0] / errors[-1]) / math.log(resolutions[0] / resolutions[-1])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("geometry", "decaying", "min_order"),
    [
        ("box", False, 1.6),  # steady, straight boundary → clean O(h²)
        ("box", True, 1.6),  # decaying, straight boundary → clean O(h²)
        ("disk", False, 1.2),  # steady, curved boundary → P1 ~O(h^1.5)
        ("disk", True, 1.2),  # decaying, curved boundary → P1 ~O(h^1.5)
    ],
    ids=["box-steady", "box-decaying", "disk-steady", "disk-decaying"],
)
def test_fsi_co_moving_species_transport_holds_its_order(geometry: str, decaying: bool, min_order: float) -> None:
    """The FSI co-moving species ALE transport holds its P1 spatial order against a manufactured solution
    (O(h²) on the box, ~O(h^1.5) on the curved disk) — a wrong-but-conservative operator would collapse it."""
    errors = [_linf_error_at(h, geometry=geometry, decaying=decaying) for h in _RESOLUTIONS_H]
    order = _order(errors, _RESOLUTIONS_H)
    assert order > min_order, f"[{geometry}, decaying={decaying}] order {order:.2f} (errors {errors}) too low"


@pytest.mark.integration
def test_fsi_species_mms_discriminates_the_dilution_term() -> None:
    """Negative control giving the order tests teeth: drop the dilution's contribution from the manufactured
    forcing while the real operator keeps `c ∇·v`. u* is then not the exact solution, so the error must NOT
    converge — a mis-scaled/missing/sign-flipped dilution in the real code cannot masquerade as passing."""
    two = [0.1, 0.05]  # two coarse resolutions suffice to show non-convergence (cheap)
    errors = [_linf_error_at(h, dilution_in_source=False) for h in two]
    order = _order(errors, two)
    assert order < 0.5, f"dilution negative control converged (order {order:.2f}, errors {errors}) — no teeth"
