"""Analytic subvolumes that touch the box (#187).

VCell models often cut a cell with the domain box: a half cell on a symmetry plane, a slab of cytoplasm
under a coverslip, a dendrite leaving the box. The marching path embeds each subvolume's boundary as a
closed curve / surface strictly inside the box, so these geometries go through the label pipeline the
image path uses: the priority rasterization on an ``h`` lattice, SurfaceNets boundaries closed by the box
faces, and a projection onto the exact implicit functions. The checks:

- **the shapes are the analytic ones:** a half disk / half ball on a box face has the exact area, volume and
  membrane measure to second order in ``h``, and the membrane vertices lie on the circle;
- **the box faces close each region:** the cut face belongs to the cell, the rest to the background;
- **a slab lying on a box face** (its predicate's zero set *is* the face, as in `z >= z0`) meshes;
- **an interior shape beside a touching one** is realized too;
- **a solve on it conserves:** an exchange across the membrane of a half-cell keeps the total.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import integrate_interface_coupled
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.realize import realize, realize_interface_coupled
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    MathDescription,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _cut_cell(dim: int, radius: float = 1.5) -> GeometryDescription:
    """A disk (ball) of ``radius`` centred on the box face x = 0 of a 4-wide box: half of it is inside."""

    squared = "geom.x[0] ** 2 + (geom.x[1] - 2) ** 2" + (" + (geom.x[2] - 2) ** 2" if dim == 3 else "")
    return GeometryDescription(
        name="cut",
        dim=dim,
        extent=(4.0, 4.0, 4.0 if dim == 3 else 1.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression=f"{squared} < {radius**2}"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )


def _measure(geometry: Geometry, name: str) -> float:
    mesh = geometry.subdomains[name].mesh
    return float(fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))).real)


@pytest.mark.parametrize("h", [0.2, 0.1])
def test_a_half_disk_on_a_box_face_is_the_analytic_half_disk(h: float) -> None:
    geometry = realize(_cut_cell(2), h=h)
    # second order in h: the chords of the polygon cut inside the arc
    assert _measure(geometry, "cell") == pytest.approx(0.5 * math.pi * 1.5**2, rel=0.2 * h**2)
    assert _measure(geometry, "ec") == pytest.approx(16.0 - 0.5 * math.pi * 1.5**2, rel=0.2 * h**2)
    assert _measure(geometry, "pm") == pytest.approx(math.pi * 1.5, rel=0.05 * h**2)
    x = geometry.subdomains["pm"].mesh.geometry.x
    assert np.allclose(np.hypot(x[:, 0], x[:, 1] - 2.0), 1.5, atol=1e-9)  # projected onto the exact circle
    # the cut face closes the cell; every other face is the background's
    assert set(geometry.boundaries["x_minus"].subdomains) == {"cell", "ec"}
    for face in ("x_plus", "y_minus", "y_plus"):
        assert geometry.boundaries[face].subdomains == ("ec",)


def test_a_half_ball_on_a_box_face_is_the_analytic_half_ball() -> None:
    geometry = realize(_cut_cell(3), h=0.2)
    assert _measure(geometry, "cell") == pytest.approx(2.0 / 3.0 * math.pi * 1.5**3, rel=0.01)
    assert _measure(geometry, "pm") == pytest.approx(2.0 * math.pi * 1.5**2, rel=0.005)
    assert _measure(geometry, "cell") + _measure(geometry, "ec") == pytest.approx(64.0, rel=1e-12)
    assert set(geometry.boundaries["x_minus"].subdomains) == {"cell", "ec"}


def test_a_thin_slab_lying_on_a_box_face_meshes() -> None:
    # BioModel 120814542's geometry: a 0.1-thick disk of radius 1.6 on the bottom face of a 3 × 3 × 1.5 box,
    # so it is also cut by the four sides. Its predicate's zero set includes the bottom face (z >= -0.1 with
    # the box starting at -0.1), which the projection must not pull the interface onto; at h = 0.1 the slab
    # is one lattice cell thick.
    description = GeometryDescription(
        name="slab",
        dim=3,
        extent=(3.0, 3.0, 1.5),
        origin=(-1.5, -1.5, -0.1),
        subvolumes=(
            SubVolume(
                name="slab",
                type="analytic",
                expression="(geom.x[2] >= -0.1) && (geom.x[2] <= 0.0) && (geom.x[0] ** 2 + geom.x[1] ** 2 < 1.6 ** 2)",
            ),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="ec", outside="slab"),),
    )
    n = 2001
    grid = np.linspace(-1.5, 1.5, n)
    footprint = float(np.mean(np.add.outer(grid**2, grid**2) < 1.6**2)) * 9.0  # the disk clipped by the square
    geometry = realize(description, h=0.1)
    assert _measure(geometry, "slab") == pytest.approx(0.1 * footprint, rel=0.02)
    assert _measure(geometry, "slab") + _measure(geometry, "ec") == pytest.approx(13.5, rel=1e-12)


def test_an_interior_shape_beside_a_touching_one() -> None:
    # a nucleus strictly inside the box next to a half cell on x = 0: both through the label path
    description = GeometryDescription(
        name="two",
        dim=2,
        extent=(4.0, 4.0, 1.0),
        subvolumes=(
            SubVolume(name="nucleus", type="analytic", expression="(geom.x[0] - 3) ** 2 + (geom.x[1] - 2) ** 2 < 0.25"),
            SubVolume(name="cell", type="analytic", expression="geom.x[0] ** 2 + (geom.x[1] - 2) ** 2 < 2.25"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="pm", inside="cell", outside="ec"),
            SurfaceClass(name="ne", inside="nucleus", outside="ec"),
        ),
    )
    geometry = realize(description, h=0.1)
    assert _measure(geometry, "nucleus") == pytest.approx(math.pi * 0.25, rel=5e-3)
    assert _measure(geometry, "cell") == pytest.approx(0.5 * math.pi * 2.25, rel=5e-3)
    assert _measure(geometry, "ne") == pytest.approx(2.0 * math.pi * 0.5, rel=5e-3)


def _pde(variable: str, subdomain: str, ic: str) -> TemplateEquation:
    return TemplateEquation(
        template="bulk_radv_diff",
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"diffusion": "1.0"},
        initial_condition=ic,
    )


def test_an_exchange_across_a_half_cells_membrane_conserves() -> None:
    # c in the half cell, u outside, exchanging across the membrane: the total is conserved, and the cell's
    # cut face (on the box) is a no-flux wall like the rest of the box
    md = MathDescription(
        geometry="cut",
        subdomains=[
            Subdomain(name="cell", kind="volume"),
            Subdomain(name="ec", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[Variable(name="c", subdomain="cell"), Variable(name="u", subdomain="ec")],
        equations=[_pde("c", "cell", "1.0 + geom.x[1]"), _pde("u", "ec", "0.0")],
        boundary_conditions=[
            BCInterfaceFlux(variable="c", boundary="pm", expression="0.5 * (u - c)"),
            BCInterfaceFlux(variable="u", boundary="pm", expression="0.5 * (c - u)"),
        ],
    )
    geometry = realize_interface_coupled(
        _cut_cell(2),
        inner_subdomain="cell",
        outer_subdomain="ec",
        membrane_subdomain="pm",
        interface="pm",
        h=0.2,
    )

    def total(fields: dict[str, fem.Function]) -> float:
        return sum(
            float(fem.assemble_scalar(fem.form(f * ufl.dx(domain=f.function_space.mesh))).real) for f in fields.values()
        )

    start = integrate_interface_coupled(md, geometry, t_final=1e-9)
    end = integrate_interface_coupled(md, geometry, t_final=1.0, rtol=1e-8, atol=1e-11)
    assert start.fields is not None and end.fields is not None
    assert total(end.fields) == pytest.approx(total(start.fields), rel=1e-7)
    assert total({"u": end.fields["u"]}) > 0.1 * total(start.fields)  # it did cross
