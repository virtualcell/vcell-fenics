"""Verification of Approach-A bulk mesh-motion (harmonic extension).

A prescribed velocity on a moving *bulk* (codim-0) subdomain only defines the
*boundary* motion; the interior is a mesh-bookkeeping device, so its nodes are
moved by the harmonic extension of the boundary displacement (∇²d = 0, d = dt·v on
∂Ω) rather than dragged by the raw velocity formula. `_MeshMotion` selects this for
codim-0 meshes; a codim-1 membrane (all nodes on the boundary) keeps the original
direct-interpolation move. The checks:

1. **Boundary follows the prescribed motion exactly** — boundary nodes are displaced
   by dt·v (the Dirichlet data of the extension).
2. **Affine motion is exact** — a uniform expansion (velocity = x, an affine/harmonic
   field) scales the whole mesh uniformly, so the cell-size ratio is invariant. This
   is the analytical exactness check (affine fields are harmonic).
3. **The interior is filled, not dragged** — a velocity singular in the interior
   (x/r(x) at the centre) still produces a clean, valid moving mesh, where a naive
   interior drag would inject the singularity.
4. **Quality stays valid under non-uniform motion** — cells stay positive (no
   inversion) and finite over a non-affine boundary motion.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import DiscreteProblem, assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml


def _model(velocity: str) -> str:
    return f"""
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "0.05" }}
      initial_condition: "1.0"
"""


def _problem(velocity: str, *, dt: float = 0.1, h: float = 0.2) -> DiscreteProblem:
    geom = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=h)
    return assemble(load_yaml(_model(velocity)), geom, dt=dt)


def _radii(dp: DiscreteProblem) -> np.ndarray:
    return np.asarray(np.linalg.norm(dp.V.mesh.geometry.x[:, :2], axis=1))


def _cell_volumes(dp: DiscreteProblem) -> np.ndarray:
    form = fem.form(ufl.TestFunction(fem.functionspace(dp.V.mesh, ("DG", 0))) * ufl.dx)
    return np.asarray(fem.assemble_vector(form).array)


# ---------------------------------------------------------------------------
# 1. boundary follows the prescribed motion exactly
# ---------------------------------------------------------------------------


def test_boundary_follows_prescribed_motion() -> None:
    dp = _problem("geom.x", dt=0.1)  # velocity = position ⇒ each node moves dt·x outward
    boundary = _radii(dp) > 0.99  # nodes that start on the unit-circle boundary

    for _ in range(5):
        dp.step()

    # Uniform radial growth: after 5 steps of factor (1+dt), boundary radius = 1.1^5.
    assert np.allclose(_radii(dp)[boundary], 1.1**5, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. affine (harmonic) motion is exact — uniform scaling, ratio invariant
# ---------------------------------------------------------------------------


def test_affine_motion_preserves_cell_ratio_exactly() -> None:
    dp = _problem("geom.x", dt=0.1)
    vols = _cell_volumes(dp)
    ratio_before = vols.max() / vols.min()

    for _ in range(8):
        dp.step()

    vols = _cell_volumes(dp)
    ratio_after = vols.max() / vols.min()
    # velocity x is affine hence harmonic: the extension reproduces it in the
    # interior, so the mesh scales uniformly and the ratio is unchanged.
    assert ratio_after == pytest.approx(ratio_before, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. the interior is harmonically filled, not dragged by the raw velocity
# ---------------------------------------------------------------------------


def test_harmonic_fill_smooths_the_interior_instead_of_dragging() -> None:
    # velocity x/r(x) is unit-radial everywhere — a naive interior drag would move
    # *every* node outward by the full dt, including the centre (in a near-arbitrary
    # direction, since the unit radial is undefined there). The harmonic extension
    # instead fills the interior from the boundary: on the unit disk x/r(x) = x on
    # ∂Ω, so the smooth fill is dt·x and a node's displacement scales with its radius
    # — the centre barely moves while the boundary moves by dt.
    dt = 0.05
    dp = _problem("geom.x / geom.radius", dt=dt)
    xy = dp.V.mesh.geometry.x[:, :2]
    centre = int(np.argmin(np.linalg.norm(xy, axis=1)))  # node nearest the origin
    edge = int(np.argmax(np.linalg.norm(xy, axis=1)))  # a boundary node
    c0, e0 = xy[centre].copy(), xy[edge].copy()
    r_centre = float(np.linalg.norm(c0))

    dp.step()

    xy = dp.V.mesh.geometry.x[:, :2]
    centre_disp = float(np.linalg.norm(xy[centre] - c0))
    edge_disp = float(np.linalg.norm(xy[edge] - e0))

    assert edge_disp == pytest.approx(dt, rel=1e-3)  # boundary follows the prescribed dt·v
    assert centre_disp == pytest.approx(dt * r_centre, abs=1e-4)  # smooth fill dt·x, not a unit drag
    assert centre_disp < 0.5 * dt  # ≪ the full-dt drag a naive interior move would apply
    assert np.all(np.isfinite(dp.V.mesh.geometry.x))
    assert float(_cell_volumes(dp).min()) > 0.0


# ---------------------------------------------------------------------------
# 4. non-uniform boundary motion keeps the mesh valid
# ---------------------------------------------------------------------------


def test_nonuniform_motion_keeps_cells_valid() -> None:
    # Angle-dependent radial speed: a non-affine boundary motion that distorts the
    # mesh. Harmonic extension keeps the interior valid (no inverted cells).
    dp = _problem("(1.0 + 0.5 * cos(2 * geom.azimuth)) * geom.x / geom.radius", dt=0.03)

    for _ in range(10):
        dp.step()

    vols = _cell_volumes(dp)
    assert np.all(np.isfinite(dp.V.mesh.geometry.x))
    assert float(vols.min()) > 0.0  # no collapsed or inverted cell


# ---------------------------------------------------------------------------
# 5. translation — a co-moving bulk transports rigidly (the discriminator baseline)
# ---------------------------------------------------------------------------

_TRANSLATING_BULK = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: prescribed, velocity: "[0.5, 0.0]" } }
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.0" }
      initial_condition: "1.0 + 0.3*geom.x[0]"
"""


def test_translation_transports_a_comoving_bulk_rigidly() -> None:
    # A constant boundary velocity [0.5, 0] has a *constant* (hence harmonic) extension,
    # so the whole bulk mesh translates rigidly. With the medium co-moving (v = mesh
    # velocity, the closed-cell regime), ∇·v = 0 ⇒ no spurious dilution: area and ∫_Ω c
    # are conserved and each material node carries its value. This is the volume baseline
    # the Eulerian (static-medium) case will be discriminated against — there a lab-frame
    # field would instead need the (u − v_mesh)·∇c convective term.
    geom = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.12)
    dp = assemble(load_yaml(_TRANSLATING_BULK), geom, dt=0.01)
    mesh = dp.V.mesh

    def _area() -> float:
        local = fem.assemble_scalar(fem.form(1.0 * dp.dx))
        return float(mesh.comm.allreduce(local, op=MPI.SUM))

    center0 = mesh.geometry.x[:, :2].mean(axis=0)
    c0 = dp.unknown.x.array.copy()
    area0, mass0 = _area(), dp.total_mass()
    for _ in range(50):
        dp.step()

    center1 = mesh.geometry.x[:, :2].mean(axis=0)
    assert center1[0] - center0[0] == pytest.approx(0.5 * 0.01 * 50)  # rigid shift by v·t
    assert abs(center1[1] - center0[1]) < 1e-12
    assert _area() == pytest.approx(area0, rel=1e-12)  # rigid: area unchanged
    assert dp.total_mass() == pytest.approx(mass0, rel=1e-12)  # no spurious dilution
    assert np.abs(dp.unknown.x.array - c0).max() < 1e-10  # each material node keeps its value


# ---------------------------------------------------------------------------
# 6. chemistry-coupled velocity — the substrate speed is a function of the field
# ---------------------------------------------------------------------------


def _coupled_model(velocity: str, ic: str, *, source: str | None = None) -> str:
    """A bulk moving subdomain whose prescribed velocity references the species `c`.
    Diffusion 0 so a uniform field stays uniform (the velocity then stays spatially
    constant and the bulk translates rigidly), making the displacement analytic."""
    terms = '{ diffusion: "0.0"' + (f', source: "{source}"' if source else "") + " }"
    return f"""
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {terms}
      initial_condition: "{ic}"
"""


def _translation_x(model: str, *, steps: int, dt: float = 0.01) -> float:
    geom = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.15)
    dp = assemble(load_yaml(model), geom, dt=dt)
    x0 = float(dp.V.mesh.geometry.x[:, 0].mean())
    for _ in range(steps):
        dp.step()
    return float(dp.V.mesh.geometry.x[:, 0].mean()) - x0


def test_chemistry_coupled_velocity_scales_with_the_field() -> None:
    # velocity = [0.5·c, 0]: the substrate speed is a function of the species field. A uniform field
    # (D = 0, no reaction → stays uniform) gives a constant speed 0.5·c0, so the bulk translates rigidly
    # by 0.5·c0·T. Doubling the field doubles the displacement — the decisive proof the motion reads the
    # *current chemistry*, not a fixed space/time expression (which is all the backend evaluated before).
    steps, dt = 20, 0.01
    d1 = _translation_x(_coupled_model("[0.5*c, 0.0]", "1.0"), steps=steps, dt=dt)
    d2 = _translation_x(_coupled_model("[0.5*c, 0.0]", "2.0"), steps=steps, dt=dt)

    assert d1 == pytest.approx(0.5 * 1.0 * dt * steps, rel=1e-6)  # 0.5·c0·T, c0 = 1
    assert d2 == pytest.approx(0.5 * 2.0 * dt * steps, rel=1e-6)  # 0.5·c0·T, c0 = 2
    assert d2 == pytest.approx(2.0 * d1, rel=1e-9)  # field value drives the speed


def test_chemistry_coupled_velocity_tracks_the_evolving_field() -> None:
    # Add first-order decay (source = −k·c): the uniform field decays, c(t) = c0·e^(−k t), so the
    # velocity 0.5·c decays with it and the bulk *slows*. The displacement is then strictly less than the
    # no-decay run — proof the velocity is re-evaluated against the field each step, not frozen at t = 0.
    steps, dt = 30, 0.01
    geom = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.15)
    dp = assemble(load_yaml(_coupled_model("[0.5*c, 0.0]", "1.0", source="-4.0*c")), geom, dt=dt)
    x0 = float(dp.V.mesh.geometry.x[:, 0].mean())
    # Discrete explicit-lag integral: each step advances by dt·0.5·cⁿ, with cⁿ the (uniform) field
    # before the step; backward Euler decays it as cⁿ⁺¹ = cⁿ/(1 + k·dt).
    c, expected = 1.0, 0.0
    for _ in range(steps):
        expected += dt * 0.5 * c
        c /= 1.0 + 4.0 * dt
        dp.step()
    moved = float(dp.V.mesh.geometry.x[:, 0].mean()) - x0

    assert dp.unknown.x.array.mean() < 0.5  # the field decayed (c0 = 1 → < 0.5 after k·T = 1.2)
    assert moved == pytest.approx(expected, rel=1e-5)  # matches the integral of the *evolving* speed
    assert moved < 0.5 * 1.0 * dt * steps  # strictly less than the undamped (constant-field) run
