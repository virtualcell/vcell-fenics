"""Species on a moving membrane and on both sides of it: the multi-compartment solver with one moving membrane
(ADR 014). There is no reference solver for a membrane that carries state while it moves, so the checks are
exact solutions and invariants that tell the conventions apart:

- **uniform stretch:** a uniform membrane density on an expanding circle is `ρ₀ |Γ₀| / |Γ(t)|`, both totals kept;
- **transport relative to sliding nodes:** a membrane flowing along itself on a fixed circle, `v = A sin θ t̂`,
  changes its species as `∂ρ/∂t = −∂_s(ρ v)`, while the slide keeps the nodes in place — so the mesh velocity
  and the material velocity differ, and only the right one moves the species;
- **the bulk at rest:** a rigidly rotating membrane rotates the inside mesh, but a linear profile stays put;
- **the outside compartment at rest:** swept against the fixed box, a well-mixed outside species keeps its
  amount, its concentration `M / |Ω_out(t)|`;
- **agreement with ADR 013's path:** species inside a translating front only, on both solvers;
- **the slide** changes nothing physical under a normal-only front.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend.multi_compartment import (
    MultiCompartmentGeometry,
    MultiCompartmentResult,
    integrate_multi_compartment,
    realize_multi_compartment,
    species_mass,
)
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_R0 = 0.5


def _cell() -> GeometryDescription:
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {_R0**2}"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )


def _model(velocity: str, species: dict[str, tuple[str, str, str]]) -> str:
    """``species``: name → (subdomain, diffusion, initial condition)."""
    variables = "\n".join(f"    - {{ name: {n}, subdomain: {sub} }}" for n, (sub, _, _) in species.items())
    equations = "\n".join(
        f"    - {{ template: {'surface_pde_with_dilution' if sub == 'pm' else 'bulk_radv_diff'}, variable: {n}, "
        f'subdomain: {sub}, temporality: time_dependent, terms: {{ diffusion: "{d}" }}, initial_condition: "{ic}" }}'
        for n, (sub, d, ic) in species.items()
    )
    return f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: cyto, kind: volume }}
    - {{ name: ext, kind: volume }}
    - {{ name: pm, kind: surface, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
{variables}
  equations:
{equations}
"""


def _measure(mesh: object) -> float:
    return float(np.real(fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))))


def _run(
    velocity: str, species: dict[str, tuple[str, str, str]], *, h: float = 0.06, t: float = 1.0, strides: int = 20
) -> tuple[MultiCompartmentGeometry, MultiCompartmentResult]:
    compartments = {sub for sub, _, _ in species.values() if sub != "pm"} | {"cyto"}
    geometry = realize_multi_compartment(_cell(), h=h, compartments=compartments)
    ends = [t * k / strides for k in range(1, strides + 1)]
    result = integrate_multi_compartment(
        load_yaml(_model(velocity, species)), geometry, t_final=t, motion_membrane="pm", stride_ends=ends
    )
    return geometry, result


def test_a_uniform_stretch_dilutes_the_membrane_species_exactly() -> None:
    geometry = realize_multi_compartment(_cell(), h=0.06, compartments={"cyto"})
    length0, area0 = _measure(geometry.membranes["pm"].mesh), _measure(geometry.compartments["cyto"].mesh)
    species = {"u": ("cyto", "1.0", "1.0"), "rho": ("pm", "0.1", "1.0")}
    result = integrate_multi_compartment(
        load_yaml(_model("0.2 * geom.normal", species)),
        geometry,
        t_final=1.0,
        motion_membrane="pm",
        stride_ends=[0.05 * k for k in range(1, 21)],
    )
    length = _measure(geometry.membranes["pm"].mesh)
    assert length / length0 == pytest.approx((_R0 + 0.2) / _R0, rel=1e-3)
    rho = result.fields["rho"].x.array
    assert rho.mean() == pytest.approx(length0 / length, rel=1e-3)  # ρ₀ |Γ₀| / |Γ(t)|
    assert rho.max() - rho.min() < 5e-3
    assert species_mass(result, "rho") == pytest.approx(length0, rel=1e-12)
    assert species_mass(result, "u") == pytest.approx(area0, rel=1e-12)


def test_a_membrane_flowing_along_itself_carries_its_species_relative_to_the_sliding_nodes() -> None:
    # v = A sin θ t̂ on a circle of radius R: the circle stays put (to O(h²) per stride) and the slide keeps the
    # nodes near their places, while the species is advected and diluted, ∂ρ/∂t = −∂_s(ρ v) = −(A/R) cos θ at
    # first; so ρ(t) ≈ 1 − t (A/R) cos θ. Transport by the mesh velocity instead would leave ρ = 1.
    amplitude, t = 0.2, 0.2
    velocity = f"{amplitude} * (geom.x[1] / geom.radius) * [-geom.x[1] / geom.radius, geom.x[0] / geom.radius]"
    _, result = _run(velocity, {"rho": ("pm", "1e-3", "1.0"), "u": ("cyto", "1.0", "1.0")}, t=t, strides=10)
    rho = result.fields["rho"]
    xy = rho.function_space.tabulate_dof_coordinates()[:, :2]
    cos = xy[:, 0] / np.linalg.norm(xy, axis=1)
    expected = 1.0 - t * (amplitude / _R0) * cos
    assert np.abs(rho.x.array - expected).max() < 0.01  # the effect itself is up to 0.08
    assert np.ptp(rho.x.array) > 0.12  # and it is there


def test_a_rigidly_rotating_membrane_leaves_the_bulk_at_rest() -> None:
    # the domain does not change while the inside mesh rotates with the membrane: u = x at rest must stay x
    # (riding the mesh would rotate it). Checked away from the membrane's boundary layer.
    omega, t = 1.0, 0.3
    _, result = _run(
        f"{omega} * [-geom.x[1], geom.x[0]]", {"u": ("cyto", "1e-3", "geom.x[0]"), "rho": ("pm", "0.1", "1.0")}, t=t
    )
    u = result.fields["u"]
    xy = u.function_space.tabulate_dof_coordinates()[:, :2]
    interior = np.linalg.norm(xy, axis=1) < 0.6 * _R0
    assert interior.sum() > 20
    # 2.3e-3 measured: the splitting error of moving the mesh in strides
    assert np.abs(u.x.array[interior] - xy[interior, 0]).max() < 5e-3
    # riding the mesh would have left each node with its initial x: off by up to r ω t ≈ 0.09
    initial_x = xy[interior, 0] * math.cos(omega * t) + xy[interior, 1] * math.sin(omega * t)
    assert np.abs(u.x.array[interior] - initial_x).max() > 0.02


def test_the_outside_species_is_swept_against_the_fixed_box() -> None:
    # an expanding membrane in a closed box: the outside compartment shrinks, its well-mixed species keeps its
    # amount, and its concentration rises as M / |Ω_out(t)|
    geometry, result = _run("0.2 * geom.normal", {"e": ("ext", "10.0", "1.0"), "rho": ("pm", "0.1", "1.0")})
    outside = _measure(geometry.compartments["ext"].mesh)
    amount = species_mass(result, "e")
    assert amount == pytest.approx(4.0 - math.pi * _R0**2, rel=2e-2)  # the initial amount (polygonal circle)
    e = result.fields["e"].x.array
    assert e.mean() == pytest.approx(amount / outside, rel=1e-2)
    assert e.mean() > 1.15  # concentrated: |Ω_out| fell from 3.21 to 2.46


def test_the_inside_species_matches_the_single_mesh_moving_path() -> None:
    # ADR 013's single-mesh path (cross-validated against mbsolver) and this solver on the same front and model
    from vcell_fenics.backend.assemble import assemble
    from vcell_fenics.backend.discrete import DiscreteProblem
    from vcell_fenics.backend.realize import realize

    species = {"u": ("cyto", "0.1", "geom.x[0] + 1.0")}
    velocity = "[0.5, 0.0]"
    _, result = _run(velocity, species, h=0.06, t=0.4, strides=40)
    md = load_yaml(_model(velocity, species))
    problem = assemble(md, realize(_cell(), h=0.06), dt=0.01)
    assert isinstance(problem, DiscreteProblem)
    for k in range(1, 41):
        problem.set_time(k * 0.01)
        problem.step()
    ours = result.fields["u"]
    theirs = problem.unknown
    xy = ours.function_space.tabulate_dof_coordinates()[:, :2]
    xy_theirs = theirs.function_space.tabulate_dof_coordinates()[:, :2]
    from scipy.spatial import cKDTree

    distance, index = cKDTree(xy_theirs).query(xy)
    same = distance < 1e-9  # the two realizations mesh the disk alike; compare at their common nodes
    assert same.mean() > 0.9
    assert np.abs(ours.x.array[same] - theirs.x.array[index[same]]).max() < 2e-2
    assert species_mass(result, "u") == pytest.approx(
        float(np.real(fem.assemble_scalar(fem.form(theirs * ufl.dx)))), rel=1e-6
    )


def test_the_slide_changes_nothing_physical_under_a_normal_front(monkeypatch: pytest.MonkeyPatch) -> None:
    # a front retracting on one side: the membrane species (non-uniform) is the same with the slide on and off
    velocity = "-0.3 * (1 + 2 * geom.x[0]) * geom.normal"
    species = {"rho": ("pm", "0.05", "1.0 + geom.x[1]"), "u": ("cyto", "1.0", "1.0")}
    geometry_on, on = _run(velocity, species, t=0.4, strides=40)

    from vcell_fenics.backend import moving_partition

    # no slide: each node stays where the front's velocity moves it
    monkeypatch.setattr(moving_partition, "_slid_to_fractions", lambda points, fractions: points)
    geometry_off, off = _run(velocity, species, t=0.4, strides=40)
    assert _measure(geometry_on.membranes["pm"].mesh) == pytest.approx(
        _measure(geometry_off.membranes["pm"].mesh), rel=2e-3
    )
    # the slide keeps the membrane's amount (no source) to 1e-7; without it the gathered nodes lose 1.1e-4
    assert species_mass(off, "rho") == pytest.approx(species_mass(on, "rho"), rel=5e-4)
    # the same field: sampled at the nearest node of the other, the two agree to discretization error
    from scipy.spatial import cKDTree

    a, b = on.fields["rho"], off.fields["rho"]
    xa = a.function_space.tabulate_dof_coordinates()[:, :2]
    xb = b.function_space.tabulate_dof_coordinates()[:, :2]
    distance, index = cKDTree(xb).query(xa)
    near = distance < 0.02
    assert near.mean() > 0.5
    assert np.abs(a.x.array[near] - b.x.array[index[near]]).max() < 3e-2


# --- the membrane-coupled solver's prescribed-motion tests, on this solver (ADR 014 decision 7) ----------------
# The same receptor model (a membrane receptor capturing ligand from both compartments, total ligand conserved)
# and reacting model as `test_backend_membrane_coupled.py`, with the membrane's velocity in the model's motion and
# both compartments' material at rest, which the membrane-coupled solver read as co-moving with the mesh.

_EXPANDING = "[0.3 * geom.x[0], 0.3 * geom.x[1]]"


def _moving(md: object, velocity: str) -> object:
    from dataclasses import replace

    from vcell_fenics.formalism.schema import MathDescription, MotionPrescribedVelocity

    assert isinstance(md, MathDescription)
    subdomains = [
        replace(s, motion=MotionPrescribedVelocity(velocity=velocity)) if s.name == "pm" else s for s in md.subdomains
    ]
    return replace(md, subdomains=subdomains)


def _solve(
    md: object, *, t: float = 0.3, strides: int = 10, h: float = 0.08
) -> tuple[MultiCompartmentGeometry, MultiCompartmentResult, float]:
    from vcell_fenics.formalism.schema import MathDescription

    assert isinstance(md, MathDescription)
    geometry = realize_multi_compartment(_cell(), h=h, compartments={"cyto", "ext"})
    area0 = _measure(geometry.compartments["cyto"].mesh)
    result = integrate_multi_compartment(
        md, geometry, t_final=t, motion_membrane="pm", stride_ends=[t * k / strides for k in range(1, strides + 1)]
    )
    return geometry, result, area0


def _ligand(result: MultiCompartmentResult) -> float:
    return sum(species_mass(result, name) for name in ("L_in", "L_out", "R"))


def _initial_ligand() -> float:
    # L_in = L_out = 1 over the box (4), R = 0: the initial total does not depend on the mesh but for the
    # polygonal membrane, which both compartments share, so their sum is the box's area exactly
    return 4.0


@pytest.mark.parametrize("velocity", [_EXPANDING, "[-0.3 * geom.x[0], -0.3 * geom.x[1]]"])
def test_total_ligand_is_conserved_while_the_membrane_moves(velocity: str) -> None:
    from tests.test_backend_membrane_coupled import _model as receptor_model

    geometry, result, area0 = _solve(_moving(receptor_model(), velocity))
    assert abs(_measure(geometry.compartments["cyto"].mesh) / area0 - 1.0) > 0.15  # deformed substantially
    assert species_mass(result, "R") > 0.1  # the receptor captured ligand
    assert _ligand(result) == pytest.approx(_initial_ligand(), rel=1e-12)


def test_without_binding_each_field_keeps_its_amount() -> None:
    from tests.test_backend_membrane_coupled import _model as receptor_model

    geometry, result, area0 = _solve(_moving(receptor_model(kon=0.0, r_ic="1.0"), _EXPANDING))
    assert species_mass(result, "L_in") == pytest.approx(area0, rel=1e-12)
    assert species_mass(result, "L_out") == pytest.approx(4.0 - area0, rel=1e-12)
    # R started at 1 on the membrane, so its amount is the initial length; v = 0.3 x scales lengths by e^(0.3 t)
    initial_length = species_mass(result, "R")
    assert initial_length == pytest.approx(2.0 * math.pi * _R0, rel=1e-2)
    assert _measure(geometry.membranes["pm"].mesh) == pytest.approx(initial_length * math.exp(0.09), rel=2e-3)


@pytest.mark.parametrize("strides", [3, 10, 30])
def test_ligand_is_conserved_to_round_off_at_any_stride(strides: int) -> None:
    # each mesh move carries every species' nodal moments exactly (ADR 015), so the exchange across the moving
    # membrane leaves the total unchanged however coarse the strides (the swept-rate scheme before it drifted
    # at first order in the stride)
    from tests.test_backend_membrane_coupled import _model as receptor_model

    _, result, _ = _solve(_moving(receptor_model(), _EXPANDING), strides=strides)
    assert _ligand(result) == pytest.approx(_initial_ligand(), rel=1e-12)


def test_stiff_binding_is_captured_while_the_membrane_moves() -> None:
    from tests.test_backend_membrane_coupled import _model as receptor_model

    _, result, _ = _solve(_moving(receptor_model(kon=1.0, rmax=10.0), _EXPANDING))
    assert species_mass(result, "R") > 2.0
    assert _ligand(result) == pytest.approx(_initial_ligand(), rel=1e-12)


def test_a_nonlinear_bulk_reaction_keeps_its_pool_while_the_membrane_moves() -> None:
    # A + B → C at rate kf·A·B in the cytoplasm, the membrane expanding: the pool A + C holds (no flux through
    # the membrane), the reaction runs, all on the moving partition
    from tests.test_backend_membrane_coupled import _reacting_model

    model = _reacting_model(src_a="-kf*A*B", src_b="-kf*A*B", src_c="kf*A*B")
    geometry, result, area0 = _solve(_moving(model, "[0.25 * geom.x[0], 0.25 * geom.x[1]]"), t=0.6, strides=24)
    assert abs(_measure(geometry.compartments["cyto"].mesh) / area0 - 1.0) > 0.15
    assert species_mass(result, "C") > 0.05
    assert species_mass(result, "A") + species_mass(result, "C") == pytest.approx(area0, rel=1e-10)


def test_a_motion_that_tangles_the_partition_stops_the_run() -> None:
    # the membrane swells past the fixed box: the cells between them turn inside out, which an assembled
    # |det J| volume cannot show; with no remeshing (ADR 014 decision 6) the run must stop, not continue
    from vcell_fenics.backend.discrete import MeshQualityError

    with pytest.raises(MeshQualityError, match="inverted or collapsed"):
        _run("2.0 * geom.normal", {"u": ("cyto", "1.0", "1.0"), "rho": ("pm", "0.1", "1.0")}, h=0.12, strides=10)
