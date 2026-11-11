"""A moving partition carried through remeshes (ADR 016 decisions 2, 4 and 6): a move that would tangle the mesh
is undone bitwise; a cell shrunk below a quarter of its as-built area triggers a remesh; the solver stops for one on
a valid mesh; a freshly remeshed mesh that tangles on its first move means the front cannot advance; and a run
continues through remeshes in bundle segments with every species' total kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dolfinx import fem

from vcell_fenics.backend.discrete import CellOrientation, MeshTangled
from vcell_fenics.backend.moving_partition import FrontCannotAdvance, MovingPartition
from vcell_fenics.backend.multi_compartment import (
    MultiCompartmentGeometry,
    integrate_multi_compartment,
    realize_multi_compartment,
)
from vcell_fenics.cli import main
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import MathDescription
from vcell_fenics.results import Bundle

_GEOMETRY = """
geometry_description:
  name: cell
  dim: 2
  extent: [2.0, 2.0, 1.0]
  origin: [-1.0, -1.0, 0.0]
  subvolumes:
    - { name: cyto, type: analytic, expression: "geom.x[0]**2 + geom.x[1]**2 < 0.25" }
    - { name: ext, type: analytic, expression: "1.0" }
  surfaces:
    - { name: pm, inside: cyto, outside: ext }
"""


def _math(omega: float) -> str:
    """A disk membrane rotating rigidly at `omega`: it shears the outside compartment against the fixed box until
    its cells tangle, but never comes near contact. A species in each compartment and one on the membrane."""
    ic = {"u": "1.0 + geom.x[0]", "w": "1.0 + geom.x[1]", "rho": "1.0 + geom.x[0]"}
    return f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: cyto, kind: volume }}
    - {{ name: ext, kind: volume }}
    - {{ name: pm, kind: surface, motion: {{ kind: prescribed, velocity: "{omega} * [-geom.x[1], geom.x[0]]" }} }}
  variables:
    - {{ name: u, subdomain: cyto }}
    - {{ name: w, subdomain: ext }}
    - {{ name: rho, subdomain: pm }}
  equations:
    - {{ template: bulk_radv_diff, variable: u, subdomain: cyto, temporality: time_dependent,
        terms: {{ diffusion: "0.1" }}, initial_condition: "{ic["u"]}" }}
    - {{ template: bulk_radv_diff, variable: w, subdomain: ext, temporality: time_dependent,
        terms: {{ diffusion: "0.1" }}, initial_condition: "{ic["w"]}" }}
    - {{ template: surface_pde_with_dilution, variable: rho, subdomain: pm, temporality: time_dependent,
        terms: {{ diffusion: "0.1" }}, initial_condition: "{ic["rho"]}" }}
"""


def _cell() -> GeometryDescription:
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.25"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )


def _setup(omega: float, h: float = 0.12) -> tuple[MathDescription, MultiCompartmentGeometry]:
    return load_yaml(_math(omega)), realize_multi_compartment(_cell(), h=h, compartments={"cyto", "ext"})


def _partition(md: MathDescription, geometry: MultiCompartmentGeometry) -> MovingPartition:
    traces = {
        name: fem.Function(fem.functionspace(geometry.mesh_of(sub), ("Lagrange", 1)))
        for name, sub in (("u", "cyto"), ("w", "ext"))
    }
    return MovingPartition(md, geometry, "pm", traces=traces)


def _meshes(geometry: MultiCompartmentGeometry) -> list[object]:
    return [
        geometry.parent_mesh,
        *(c.mesh for c in geometry.compartments.values()),
        *(m.mesh for m in geometry.membranes.values()),
    ]


def test_a_move_that_would_tangle_the_mesh_is_undone_bitwise() -> None:
    md, geometry = _setup(omega=20.0)
    motion = _partition(md, geometry)
    before = [m.geometry.x.copy() for m in _meshes(geometry)]  # type: ignore[attr-defined]
    with pytest.raises(MeshTangled, match="inverted or collapsed"):
        # one explicit move of 1 rad: the membrane moves out to r = 0.71 (inside the box, no contact), while the
        # outside cells, pinned at the box, shear past inverting
        motion.advance(0.05, 0.05)
    for mesh, x in zip(_meshes(geometry), before, strict=True):
        assert np.array_equal(mesh.geometry.x, x)  # type: ignore[attr-defined]


def test_a_cell_shrunk_below_a_quarter_of_its_as_built_area_triggers_a_remesh() -> None:
    md, geometry = _setup(omega=1.0)
    motion = _partition(md, geometry)
    assert not motion.degraded()
    x = geometry.parent_mesh.geometry.x
    built = x.copy()
    x[:, :2] *= 0.6  # every cell at 0.36 of its area
    assert not motion.degraded()
    x[:, :2] = built[:, :2] * 0.45  # 0.2 of it
    assert motion.degraded()


def test_the_solver_stops_for_a_remesh_on_a_valid_mesh() -> None:
    md, geometry = _setup(omega=20.0)
    orientation = CellOrientation(geometry.parent_mesh)
    result = integrate_multi_compartment(
        md,
        geometry,
        t_final=1.0,
        motion_membrane="pm",
        stride_ends=[0.05 * k for k in range(1, 21)],
        stop_for_remesh=True,
    )
    assert result.needs_remesh
    assert 0.0 < result.final_time < 1.0
    orientation.check("after the stop")  # the mesh it stopped on is untangled: the last move was undone
    assert result.last_dt is not None and result.last_dt > 0.0


def test_a_fresh_mesh_that_tangles_on_its_first_move_means_the_front_cannot_advance() -> None:
    md, geometry = _setup(omega=40.0)  # the first half-move is 1 rad: it tangles the mesh, no contact
    initial = {}
    for name, sub in (("u", "cyto"), ("w", "ext"), ("rho", "pm")):
        f = fem.Function(fem.functionspace(geometry.mesh_of(sub), ("Lagrange", 1)))
        f.x.array[:] = 1.0
        initial[name] = f
    with pytest.raises(FrontCannotAdvance, match=r"cannot advance at t = 0\.025"):
        integrate_multi_compartment(
            md,
            geometry,
            t_final=0.05,
            motion_membrane="pm",
            stride_ends=[0.05],
            initial=initial,
            stop_for_remesh=True,
        )


def test_a_rotating_membrane_runs_through_remeshes_with_every_total_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ADR 016 decision 4: the fast rotation tangles the outside compartment twice by t = 0.75 (one remesh between a
    # stride's halves, one at a stride's start); each remesh starts a bundle segment and keeps every total
    (tmp_path / "geom.yaml").write_text(_GEOMETRY)
    (tmp_path / "math.yaml").write_text(_math(5.0))
    argv = [
        "--math",
        str(tmp_path / "math.yaml"),
        "--geometry",
        str(tmp_path / "geom.yaml"),
        "--t-final",
        "0.75",
        "--dt",
        "0.05",
        "--output-dt",
        "0.25",
        "--h",
        "0.12",
        "--out",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    log = capsys.readouterr().err
    assert log.count("remeshed the moving partition") >= 2
    assert "second half-move follows on the new mesh" in log
    bundle = Bundle.open(tmp_path / "out" / "results.fenics")
    assert len(bundle.manifest.segments) >= 3
    assert len(bundle.times) == 4
    for domain, variable in (("cyto", "u"), ("ext", "w"), ("pm", "rho")):
        totals = bundle.stats(domain, variable)[:, 1]
        assert np.allclose(totals, totals[0], rtol=1e-12, atol=0.0)
