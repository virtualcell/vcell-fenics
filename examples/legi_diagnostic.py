"""Diagnostic for the LEGI demo: does the cell actually migrate? (Spoiler: no — see README.md.)

Two panels (written to `legi_diag.png`):
  - the membrane outline at a sequence of times, each with its TRUE centre of mass (the area centroid
    ∫x dx / ∫dx, not the membrane node-mean) — they overlap as one centred circle, CoM at the origin;
  - the surface-tension Stokes velocity at a polarised time — a recirculating Marangoni flow.

Together they show the cell treadmills its membrane (tangential surface flow) without translating the
centre of mass: a pure surface-tension force balance has no net propulsion. The model's `model()` is
imported from the demo so the two stay in sync.

    .pixi/envs/dev/bin/python examples/legi_diagnostic.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from vcell_fenics.backend.geometry import make_two_bulk_membrane_geometry
from vcell_fenics.backend.interface_coupled import (
    ForceBalanceMeshMotion,
    assemble_membrane_coupled,
)
from vcell_fenics.backend.stokes import solve_incompressible_stokes_surface_tension

_HERE = Path(__file__).parent
_DT, _BASE, _ALPHA = 0.02, 0.5, 0.12


def _demo_model():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("legi_demo", _HERE / "legi_sustained_chemotaxis.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model()


def main() -> None:
    geom = make_two_bulk_membrane_geometry(
        "cell", inner="cyto", outer_subdomain="ext", membrane="pm", interface="pm",
        outer="wall", inner_radius=0.5, outer_radius=1.5, h=0.06,
    )
    motion = ForceBalanceMeshMotion(geom, tension=_BASE, dt=_DT, area_correction=True)
    problem = assemble_membrane_coupled(_demo_model(), geom, dt=_DT, motion=motion)
    cyto, mem = geom.inner_mesh, geom.membrane_mesh
    scalar_space = problem.field("a").function_space
    area_form = fem.form(fem.Constant(cyto, PETSc.ScalarType(1.0)) * ufl.dx)  # type: ignore[operator]
    coord = ufl.SpatialCoordinate(cyto)

    def outline() -> np.ndarray:  # type: ignore[type-arg]
        x = mem.geometry.x[:, :2]
        c = x.mean(axis=0)
        loop = x[np.argsort(np.arctan2(x[:, 1] - c[1], x[:, 0] - c[0]))]
        return np.vstack([loop, loop[:1]])

    def centroid() -> tuple[float, float]:  # the true CoM (area centroid)
        a = float(fem.assemble_scalar(area_form).real)
        cx = float(fem.assemble_scalar(fem.form(coord[0] * ufl.dx)).real) / a
        cy = float(fem.assemble_scalar(fem.form(coord[1] * ufl.dx)).real) / a
        return cx, cy

    outlines = [(0.0, outline(), centroid())]
    velocity = None
    for step in range(280):
        problem.step()
        motion.tension.x.array[:] = _BASE + _ALPHA * (problem.field("a").x.array - problem.field("h").x.array)
        if (step + 1) % 40 == 0:
            outlines.append(((step + 1) * _DT, outline(), centroid()))
        if step + 1 == 200:  # capture the velocity field at t = 4, when the response is polarised
            u, _ = solve_incompressible_stokes_surface_tension(
                cyto, tension=motion.tension, viscosity=1.0, screening=1.0
            )
            v1 = fem.Function(fem.functionspace(cyto, ("Lagrange", 1, (2,))))
            v1.interpolate(u)
            v_xy = v1.function_space.tabulate_dof_coordinates()[:, :2]
            v_uv = v1.x.array.reshape((-1, 2))
            s_xy = scalar_space.tabulate_dof_coordinates()[:, :2]
            response = problem.field("a").x.array - problem.field("h").x.array
            n_cells = cyto.topology.index_map(cyto.topology.dim).size_local
            cells = np.array([scalar_space.dofmap.cell_dofs(c) for c in range(n_cells)])
            velocity = (v_xy, v_uv, s_xy, response, cells, outline())

    fig, (ax_out, ax_vel) = plt.subplots(1, 2, figsize=(13, 6.2))
    for i, (t, loop, com) in enumerate(outlines):
        color = plt.cm.viridis(i / (len(outlines) - 1))
        ax_out.plot(loop[:, 0], loop[:, 1], color=color, lw=1.3, label=f"t={t:.1f}")
        ax_out.plot(com[0], com[1], "o", color=color, ms=5)
    ax_out.axvline(0, color="gray", lw=0.5)
    ax_out.axhline(0, color="gray", lw=0.5)
    ax_out.set_aspect("equal")
    ax_out.legend(fontsize=7, ncol=2)
    ax_out.grid(alpha=0.3)
    ax_out.set_title("Membrane outline + true CoM (• = area centroid) — it does not move")

    assert velocity is not None
    v_xy, v_uv, s_xy, response, cells, loop = velocity
    tri = mtri.Triangulation(s_xy[:, 0], s_xy[:, 1], cells)
    contour = ax_vel.tricontourf(tri, response, levels=20, cmap="coolwarm")
    m = np.linalg.norm(v_uv, axis=1) > 1e-9
    ax_vel.quiver(v_xy[m, 0], v_xy[m, 1], v_uv[m, 0], v_uv[m, 1], scale=0.25, width=0.004, alpha=0.8)
    ax_vel.plot(loop[:, 0], loop[:, 1], "k-", lw=1.5)
    ax_vel.set_aspect("equal")
    ax_vel.set_title(
        f"Stokes velocity at t=4 (max|u|={np.linalg.norm(v_uv, axis=1).max():.3f}) — a Marangoni circulation"
    )
    fig.colorbar(contour, ax=ax_vel, fraction=0.046, label="response a−h")

    fig.tight_layout()
    out = _HERE / "legi_diag.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"  wrote {out.name}: the cell treadmills (Marangoni surface flow) but its CoM does not translate")


if __name__ == "__main__":
    main()
