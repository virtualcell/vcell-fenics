"""The cell force balance (T9) from a MathDescription on a realized geometry (ADR 018 staging step 3a): the
disk's exact solution, the energy identity, and the refusals that need the geometry or the coefficients.

The disk (ADR 017 decision 6): one cell of radius `R` under a uniform pressure `P`, a tension `γ` and a drag
`ζ`, in a medium that is not a member. The flow is radial, `v = u(r) e_r`. With either viscous operator the
volume balance reduces to `L (u'' + u'/r − u/r²) = ζ u`, so `u = A I₁(r/ℓ)` with `ℓ² = L/ζ`:
- `symmetric`: `L = η_b + 4η/3` (the 3D deviator in planar flow), and the normal stress at `r = R` is
  `σ_rr = (2η/3)(2u' − u/r) + η_b(u' + u/r) − P`;
- `laplacian`: `L = η + η_b`, and `σ_rr = η u' + η_b(u' + u/r) − P`.
The tension's natural condition is `σ_rr = −γ/R`, which fixes `A`.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem
from dolfinx import mesh as dmesh
from scipy.special import iv

from vcell_fenics.backend.cell_force_balance import (
    LOCKING_RATIO,
    ForceBalanceError,
    assemble_cell_force_balance,
    solve_cell_force_balance,
    term_powers,
)
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.formalism import load_dict
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

R = 1.0
ETA, ETA_B, ZETA, P, GAMMA = 1.0, 0.5, 4.0, 2.0, 0.5


def _disk() -> GeometryDescription:
    return GeometryDescription(
        name="disk",
        dim=2,
        extent=(4.0, 4.0, 1.0),
        origin=(-2.0, -2.0, 0.0),
        subvolumes=(
            SubVolume(name="cyt", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {R * R}"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyt", outside="ec"),),
    )


_ONE_CELL = f"""
math_description:
  geometry: disk
  subdomains:
    - {{ name: cyt, kind: volume, motion: {{ kind: unknown, variable: v }} }}
    - {{ name: ec, kind: volume }}
    - {{ name: pm, kind: surface, motion: {{ kind: unknown, variable: v }} }}
  variables:
    - {{ name: v, subdomain: cyt, type: vector }}
  equations:
    - template: cell_force_balance
      variable: v
      subdomain: cyt
      temporality: steady_state
      volumes:
        cyt: {{ viscosity: eta, bulk_viscosity: eta_b, drag: zeta, pressure: P }}
      surfaces:
        pm: {{ tension: gamma }}
  parameters:
    - {{ name: eta, value: {ETA} }}
    - {{ name: eta_b, value: {ETA_B} }}
    - {{ name: zeta, value: {ZETA} }}
    - {{ name: P, value: {P} }}
    - {{ name: gamma, value: {GAMMA} }}
"""


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_ONE_CELL)
    return raw


def _equation(raw: dict[str, Any]) -> dict[str, Any]:
    eq: dict[str, Any] = raw["math_description"]["equations"][0]
    return eq


def _exact(operator: str) -> tuple[float, float]:
    """(A, ℓ) of `u = A I₁(r/ℓ)` (the module docstring)."""

    L = ETA_B + 4 * ETA / 3 if operator == "symmetric" else ETA + ETA_B
    ell = math.sqrt(L / ZETA)
    x = R / ell
    i1 = float(iv(1, x))
    di1 = float(iv(0, x)) - i1 / x  # I₁'(x)
    if operator == "symmetric":
        per_a = (2 * ETA / 3) * (2 * di1 / ell - i1 / R) + ETA_B * (di1 / ell + i1 / R)
    else:
        per_a = ETA * di1 / ell + ETA_B * (di1 / ell + i1 / R)
    return (P - GAMMA / R) / per_a, ell


def _snapped_disk(h: float) -> MultiCompartmentGeometry:
    """The realized disk with its membrane's vertices moved radially onto the circle `r = R`. The realizer's
    marched boundary sits up to ~1e-3 off the circle and converges only at first order in `h` (issue #249;
    1.5e-3 at h = 0.1, 7.8e-4 at h = 0.05), which would hide the solver's order behind the geometry's. The
    geometry's `sizes` and membrane submeshes are left unsnapped (unused here)."""

    geometry = realize_multi_compartment(_disk(), h=h)
    parent = geometry.parent_mesh
    facets = geometry.facet_tags.find(geometry.membranes["pm"].facet_tag)
    nodes = np.unique(dmesh.entities_to_geometry(parent, 1, facets))
    x = parent.geometry.x
    r = np.hypot(x[nodes, 0], x[nodes, 1])
    x[nodes, 0] *= R / r
    x[nodes, 1] *= R / r
    return geometry


def _error(operator: str, h: float) -> float:
    raw = _raw()
    if operator == "laplacian":
        _equation(raw)["viscous_operator"] = "laplacian"
    balance = assemble_cell_force_balance(load_dict(raw), _snapped_disk(h))
    balance.solve()
    A, ell = _exact(operator)
    mesh = balance.members.mesh
    exact_space = fem.functionspace(mesh, ("Lagrange", 3, (2,)))
    exact = fem.Function(exact_space)

    def field(x: np.ndarray) -> np.ndarray:
        r = np.maximum(np.hypot(x[0], x[1]), 1e-14)
        u = A * iv(1, r / ell)
        return np.vstack([u * x[0] / r, u * x[1] / r])

    exact.interpolate(field)
    v = balance.velocity
    error = math.sqrt(fem.assemble_scalar(fem.form(ufl.inner(v - exact, v - exact) * ufl.dx)).real)
    norm = math.sqrt(fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * ufl.dx)).real)
    return error / norm


@pytest.mark.parametrize("operator", ["symmetric", "laplacian"])
def test_a_disk_relaxes_as_the_bessel_solution_at_second_order(operator: str) -> None:
    """ADR 017 decision 6's exact solution, through the formalism, the realization and the solve."""

    coarse, fine = _error(operator, 0.05), _error(operator, 0.025)
    assert fine < 3e-4
    assert math.log2(coarse / fine) > 1.8, (coarse, fine)


def test_the_disk_expands_when_the_pressure_beats_the_tension() -> None:
    balance = solve_cell_force_balance(load_dict(_raw()), _disk(), h=0.2)
    x = ufl.SpatialCoordinate(balance.members.mesh)
    outward = fem.assemble_scalar(fem.form(ufl.dot(balance.velocity, x) * ufl.dx)).real
    assert outward > 0.0  # P = 2 > γ/R = 0.5


def test_the_energy_identity_holds_with_every_term() -> None:
    """ADR 018 decision 5, ADR 019 decision 2: the term powers on the solved `v` sum to zero, with an active
    stress, a traction that reads the normal, the Laplacian operator and the semi-implicit tension."""

    raw = _raw()
    eq = _equation(raw)
    eq["viscous_operator"] = "laplacian"
    eq["volumes"]["cyt"]["active_stress"] = "1 + 0.5 * geom.x[0]"
    eq["volumes"]["cyt"]["pressure"] = "K * (V0 - region_size(cyt))"
    eq["surfaces"]["pm"]["traction"] = "0.3 * geom.x[1] * geom.normal"
    raw["math_description"]["parameters"] += [{"name": "K", "value": 2.0}, {"name": "V0", "value": 3.0}]
    balance = solve_cell_force_balance(load_dict(raw), _disk(), h=0.2, semi_implicit_dt=0.05)
    powers = term_powers(balance.problem)
    assert set(powers) == {
        "viscous_laplacian",
        "drag",
        "cell_pressure",
        "active_stress",
        "surface_tension",
        "tension_semi_implicit",
        "surface_traction",
    }
    scale = max(abs(p) for p in powers.values())
    assert abs(sum(powers.values())) < 1e-10 * scale, powers
    # the dissipation is positive and the region_size pressure was compiled (|cyt| ≈ π, so P ≈ 2(3 − π) < 0)
    assert powers["viscous_laplacian"] > 0 and powers["drag"] > 0


def test_a_slot_that_reads_a_species_waits_for_chemistry() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "m", "subdomain": "cyt", "type": "scalar"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "m",
            "subdomain": "cyt",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    md["boundary_conditions"] = [{"kind": "neumann", "variable": "m", "boundary": "pm", "expression": "0"}]
    _equation(raw)["volumes"]["cyt"]["active_stress"] = "2 * m"
    geometry = realize_multi_compartment(_disk(), h=0.3)
    with pytest.raises(ForceBalanceError, match=r"FieldsNotYetSupported: volumes.cyt.active_stress reads \['m'\]"):
        assemble_cell_force_balance(load_dict(raw), geometry)


def test_a_tension_that_turns_negative_stops_the_solve() -> None:
    raw = _raw()
    _equation(raw)["surfaces"]["pm"]["tension"] = "1 - sim.t"
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))
    balance.solve(0.5)
    with pytest.raises(ForceBalanceError, match="NonPositiveTension: the tension of 'pm' reaches -1 at t = 2"):
        balance.solve(2.0)


def test_a_viscosity_that_vanishes_is_refused() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt"]["viscosity"] = "0.5 - geom.x[0]"
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))
    with pytest.raises(ForceBalanceError, match="NonPositiveViscosity: the viscosity of member 'cyt'"):
        balance.solve()


def test_a_large_bulk_viscosity_warns_of_locking() -> None:
    raw = _raw()
    raw["math_description"]["parameters"][1]["value"] = 2 * LOCKING_RATIO
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))
    with pytest.warns(UserWarning, match="bulk_viscosity / viscosity reaches 200"):
        balance.solve()
    raw["math_description"]["parameters"][1]["value"] = ETA_B
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        balance.solve()


# -- two cells ----------------------------------------------------------------


def _two_disks() -> GeometryDescription:
    return GeometryDescription(
        name="two",
        dim=2,
        extent=(6.0, 3.0, 1.0),
        origin=(-3.0, -1.5, 0.0),
        subvolumes=(
            SubVolume(name="c1", type="analytic", expression="(geom.x[0] + 1.5)**2 + geom.x[1]**2 < 1"),
            SubVolume(name="c2", type="analytic", expression="(geom.x[0] - 1.5)**2 + geom.x[1]**2 < 1"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(
            SurfaceClass(name="pm1", inside="c1", outside="ec"),
            SurfaceClass(name="pm2", inside="c2", outside="ec"),
        ),
    )


def _two_cells(drag_on: tuple[str, ...]) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        """
math_description:
  geometry: two
  subdomains:
    - { name: c1, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: c2, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: ec, kind: volume }
    - { name: pm1, kind: surface, motion: { kind: unknown, variable: v } }
    - { name: pm2, kind: surface, motion: { kind: unknown, variable: v } }
  variables:
    - { name: v, subdomain: [c1, c2], type: vector }
  equations:
    - template: cell_force_balance
      variable: v
      subdomain: [c1, c2]
      temporality: steady_state
      volumes:
        c1: { viscosity: "1", pressure: "1" }
        c2: { viscosity: "1", pressure: "1" }
      surfaces:
        pm1: { tension: "0.5" }
        pm2: { tension: "0.5" }
"""
    )
    for cell in drag_on:
        _equation(raw)["volumes"][cell]["drag"] = "2"
    return raw


def test_each_separate_cell_must_be_held() -> None:
    """ADR 018 decision 4: the validator sees a drag on some member; the realized geometry shows `c2` is a
    group of its own with nothing resisting its rigid motion."""

    geometry = realize_multi_compartment(_two_disks(), h=0.25)
    with pytest.raises(
        ForceBalanceError, match=r"UnconstrainedRigidMotion: nothing resists a rigid motion of \['c2'\]"
    ):
        assemble_cell_force_balance(load_dict(_two_cells(("c1",))), geometry)
    balance = assemble_cell_force_balance(load_dict(_two_cells(("c1", "c2"))), geometry)
    v = balance.solve()
    assert np.isfinite(v.x.array).all() and np.abs(v.x.array).max() > 1e-3


def test_a_membrane_held_at_zero_velocity_holds_its_cell() -> None:
    raw = _two_cells(("c1",))
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "pm2", "expression": "0"}
    ]
    del _equation(raw)["surfaces"]["pm2"]
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_two_disks(), h=0.25))
    v = balance.solve()
    held = fem.locate_dofs_topological(
        balance.problem.V, 1, balance.members.facet_tags.find(balance.geometry.membranes["pm2"].facet_tag)
    )
    assert held.size > 0
    assert np.abs(v.x.array.reshape(-1, 2)[held]).max() == 0.0


def test_a_cell_touching_the_box_must_be_held_there() -> None:
    """ADR 018 decision 4 (`MemberOnFreeBox`): the cytosol fills the box's left side."""

    description = GeometryDescription(
        name="wall",
        dim=2,
        extent=(4.0, 2.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        subvolumes=(
            SubVolume(name="cyt", type="analytic", expression="geom.x[0] < 1.5"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyt", outside="ec"),),
    )
    raw = _raw()
    raw["math_description"]["geometry"] = "wall"
    geometry = realize_multi_compartment(description, h=0.25)
    with pytest.raises(ForceBalanceError, match="MemberOnFreeBox: member 'cyt' touches the box at"):
        assemble_cell_force_balance(load_dict(raw), geometry)
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": face, "expression": "0"}
        for face in sorted(geometry.compartments["cyt"].faces)
    ]
    balance = assemble_cell_force_balance(load_dict(raw), geometry)
    v = balance.solve()
    assert np.abs(v.x.array).max() > 1e-4
    for face in geometry.compartments["cyt"].faces:
        held = fem.locate_dofs_topological(
            balance.problem.V, 1, balance.members.facet_tags.find(balance.face_tags[face])
        )
        assert held.size > 0, face
        assert np.abs(v.x.array.reshape(-1, 2)[held]).max() == 0.0, face


# -- slots that must agree (the review: the energy identity cannot see a slot's sign) --------------------------


def _solved(raw: dict[str, Any], geometry: MultiCompartmentGeometry) -> np.ndarray:
    return np.array(assemble_cell_force_balance(load_dict(raw), geometry).solve().x.array)


def test_an_active_stress_and_a_normal_traction_load_as_a_pressure() -> None:
    """`pressure: P` ≡ `active_stress: -P` ≡ a traction `P geom.normal` on the cell's membrane (a uniform `P`
    is `P∮φ·n`): each slot's sign and the normal's orientation, end to end."""

    geometry = realize_multi_compartment(_disk(), h=0.2)
    base = _solved(_raw(), geometry)
    raw = _raw()
    del _equation(raw)["volumes"]["cyt"]["pressure"]
    _equation(raw)["volumes"]["cyt"]["active_stress"] = "-P"
    assert np.allclose(_solved(raw, geometry), base, atol=1e-12)
    raw = _raw()
    del _equation(raw)["volumes"]["cyt"]["pressure"]
    _equation(raw)["surfaces"]["pm"]["traction"] = "P * geom.normal"
    assert np.allclose(_solved(raw, geometry), base, atol=1e-10)
    assert np.abs(base).max() > 1e-2


def _annulus() -> GeometryDescription:
    return GeometryDescription(
        name="annulus",
        dim=2,
        extent=(4.0, 4.0, 1.0),
        origin=(-2.0, -2.0, 0.0),
        subvolumes=(
            SubVolume(name="nuc", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.16"),
            SubVolume(name="cyt", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(
            SurfaceClass(name="ne", inside="nuc", outside="cyt"),
            SurfaceClass(name="pm", inside="cyt", outside="ec"),
        ),
    )


def test_the_normal_points_out_of_the_surfaces_inside_even_where_the_cell_is_outside() -> None:
    """On `ne` the cell is the outside, so `geom.normal` points into the cell, against the members' mesh's
    facet normal. A pressure `P` on the cytosol is a traction `P n` on `pm` and `−P n` on `ne`."""

    geometry = realize_multi_compartment(_annulus(), h=0.15)
    raw: dict[str, Any] = yaml.safe_load(
        """
math_description:
  geometry: annulus
  subdomains:
    - { name: nuc, kind: volume }
    - { name: cyt, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: ec, kind: volume }
    - { name: ne, kind: surface, motion: { kind: unknown, variable: v } }
    - { name: pm, kind: surface, motion: { kind: unknown, variable: v } }
  variables:
    - { name: v, subdomain: cyt, type: vector }
  equations:
    - template: cell_force_balance
      variable: v
      subdomain: cyt
      temporality: steady_state
      volumes:
        cyt: { viscosity: "1", drag: "4", pressure: "2" }
  parameters: []
"""
    )
    base = _solved(raw, geometry)
    del _equation(raw)["volumes"]["cyt"]["pressure"]
    _equation(raw)["surfaces"] = {"pm": {"traction": "2 * geom.normal"}, "ne": {"traction": "-2 * geom.normal"}}
    assert np.allclose(_solved(raw, geometry), base, atol=1e-10)
    assert np.abs(base).max() > 1e-2


def test_a_drag_that_is_nowhere_positive_leaves_rigid_motion_unresisted() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt"]["drag"] = "zeta * sim.t"
    balance = assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))
    with pytest.raises(
        ForceBalanceError, match=r"UnconstrainedRigidMotion: the drag of \['cyt'\] is nowhere positive at t = 0"
    ):
        balance.solve(0.0)
    assert np.isfinite(balance.solve(1.0).x.array).all()


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (("volumes", "cyt", "active_stress", "geom.normal[0]"), "NormalNotYetSupported: volumes.cyt.active_stress"),
        (("surfaces", "pm", "traction", "pn * geom.normal"), "through the parameter\\(s\\) \\['pn'\\]"),
    ],
)
def test_the_normal_is_read_only_in_a_traction(edit: tuple[str, str, str, str], match: str) -> None:
    raw = _raw()
    raw["math_description"]["parameters"].append({"name": "pn", "subdomain": "pm", "expression": "geom.normal[0]"})
    where, name, slot, text = edit
    _equation(raw)[where][name][slot] = text
    with pytest.raises(ForceBalanceError, match=match):
        assemble_cell_force_balance(load_dict(raw), realize_multi_compartment(_disk(), h=0.3))


def test_a_membrane_velocity_may_not_read_the_normal_or_a_region_variable() -> None:
    raw = _two_cells(("c1",))
    md = raw["math_description"]
    del _equation(raw)["surfaces"]["pm2"]
    md["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "pm2", "expression": "0.1 * geom.normal"}
    ]
    geometry = realize_multi_compartment(_two_disks(), h=0.25)
    with pytest.raises(ForceBalanceError, match="NormalNotYetSupported: the dirichlet BC on 'pm2'"):
        assemble_cell_force_balance(load_dict(raw), geometry)
    md["variables"].append({"name": "R", "subdomain": "c2", "type": "scalar", "space": "region"})
    md["equations"].append(
        {
            "template": "region_ode",
            "variable": "R",
            "subdomain": "c2",
            "temporality": "time_dependent",
            "terms": {"uniform_rate": "0"},
            "initial_condition": "1",
        }
    )
    md["boundary_conditions"][0]["expression"] = "[trace(R), 0]"
    with pytest.raises(ForceBalanceError, match=r"FieldsNotYetSupported: the dirichlet BC on 'pm2' reads \['R'\]"):
        assemble_cell_force_balance(load_dict(raw), geometry)


def test_a_cells_membrane_must_move_with_it_or_be_a_growing_front() -> None:
    geometry = realize_multi_compartment(_disk(), h=0.3)
    raw = _raw()
    del raw["math_description"]["subdomains"][2]["motion"]
    del _equation(raw)["surfaces"]["pm"]
    with pytest.raises(ForceBalanceError, match="surface 'pm' bounds a member, so it must have motion"):
        assemble_cell_force_balance(load_dict(raw), geometry)
    raw["math_description"]["subdomains"][2]["motion"] = {
        "kind": "prescribed",
        "velocity": "trace(v) + 0.1 * geom.normal",
    }
    v = assemble_cell_force_balance(load_dict(raw), geometry).solve()  # a growing front: the solve is the same
    assert np.isfinite(v.x.array).all()
