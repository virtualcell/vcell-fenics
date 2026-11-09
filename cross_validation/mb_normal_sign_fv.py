"""mbsolver (../pyvcell/.venv): which way does VCell's surface normal point? (ADR 013, decision 2)

A disk of cytoplasm (radius R0) in an extracellular background, whose front velocity is ``V · normal``
(the surface object's ``SurfaceNormal`` quantity, which inlines to ``normalX()``/``normalY()``). If the
normal points out of the membrane's inside compartment, the disk grows as ``R0 + V t``; if it points in,
it shrinks as ``R0 − V t``. The importer maps ``normalX()`` to the formalism's ``geom.normal[0]`` (§1.8.4:
pointing away from the region the membrane encloses) without a sign change only if it grows.

Result (2026-11-09, mesh 41×41): mean front radius 3.00, 3.24, 3.48, 3.73, 3.97 at t = 0, 0.5, 1, 1.5, 2
against ``R0 + V t`` = 3.00, 3.25, 3.50, 3.75, 4.00 — outward.

    ../pyvcell/.venv/bin/python -I cross_validation/mb_normal_sign_fv.py
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pyvcell.vcml as vc
from pyvcell.vcml import vcml_mb_simulation as mbs
from pyvcell.vcml.models import Biomodel, Model

R0, V, CX, EXTENT = 3.0, 0.5, 8.0, 16.0


@contextmanager
def _silence() -> Iterator[None]:
    """Mute the native solver's per-substep chatter (it writes to fd 1 from C++)."""
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def main() -> None:
    geo = vc.Geometry(name="square", dim=2, extent=(EXTENT, EXTENT, EXTENT), origin=(0.0, 0.0, 0.0))
    geo.add_sphere("cell", radius=R0, center=(CX, CX, 0.0))
    geo.add_background("ec")
    geo.add_surface("cell_ec_membrane", "cell", "ec")
    model = Model(name="m")
    for compartment, dim in (("cyt", 3), ("ec", 3), ("pm", 2)):
        model.add_compartment(compartment, dim=dim)
    model.add_species("u", "cyt")
    biomodel = Biomodel(name="MB", model=model)
    app = biomodel.add_application("a", geometry=geo)
    app.map_compartment("cyt", "cell")
    app.map_compartment("ec", "ec")
    app.map_compartment("pm", "cell_ec_membrane")
    app.map_species("u", init_conc="1", diff_coef=1.0)
    app.add_moving_boundary_sim(name="s", duration=2.0, output_time_step=0.5, mesh_size=(41, 41, 1))

    # The normal is a quantity of the surface object VCell generates; find its name, and enable the
    # SurfaceNormal category (pyvcell's front injection enables only SurfaceVelocity).
    match = re.search(r'SpatialObject Name="(sobj_[^"]+)" Type="Surface"', mbs.to_vcml_str(biomodel))
    assert match is not None, "no surface object generated"
    surface = match.group(1)
    inject = mbs._inject_front_kinematics

    def inject_with_normal(vcml: str, application: str, front: object) -> str:
        out = inject(vcml, application, front)  # type: ignore[arg-type]
        return re.sub(r'(QuantityCategory Name="SurfaceNormal" Enabled=")false', r"\1true", out)

    mbs._inject_front_kinematics = inject_with_normal  # type: ignore[assignment]
    app.set_moving_boundary_front(velocity_x=f"{V}*{surface}_normalX", velocity_y=f"{V}*{surface}_normalY")
    with _silence():
        result = vc.simulate_moving_boundary(biomodel, "s")

    for t, frame in zip(result.times, result.frames, strict=True):
        radius = float(np.hypot(frame.front[:, 0] - CX, frame.front[:, 1] - CX).mean())
        print(f"  t = {t:4.2f}: mean front radius {radius:.4f}  (outward {R0 + V * t:.4f}, inward {R0 - V * t:.4f})")


if __name__ == "__main__":
    main()
