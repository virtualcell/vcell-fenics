"""End-to-end smoke test for the viz pipeline.

Validates that a real solver output flows through both:
- ``write_snapshot`` → XDMF on disk (ParaView path)
- ``quick_plot`` → PNG screenshot via PyVista (in-process path)

This catches regressions in the mesh ↔ VTK conversion, dolfinx.plot.vtk_mesh,
PyVista offscreen rendering, and XDMF serialization. It is not a correctness
test for the physics — that's covered by the other test files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from dolfinx import fem
from scipy.special import j0, jn_zeros

from vcell_fenics.approaches.static import BulkPDE, create_disk
from vcell_fenics.viz import quick_plot, write_snapshot


@pytest.fixture(scope="module")
def bessel_solution() -> fem.Function:
    """A few BE steps on the Bessel eigenmode → a non-trivial Function."""
    R, D, dt, n_steps = 1.0, 0.1, 0.01, 10
    lam = jn_zeros(1, 1)[0] / R

    disk = create_disk(radius=R, h=0.1)
    pde = BulkPDE(disk.mesh, D=D, dt=dt)

    def eigenmode(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        # cast: scipy.special.j0 is opaque to mypy (follow_imports=skip);
        # at runtime it returns an ndarray.
        return cast(npt.NDArray[Any], j0(lam * r))

    pde.set_initial(eigenmode)
    for _ in range(n_steps):
        pde.step()
    return pde.c


def test_write_xdmf_snapshot(tmp_path: Path, bessel_solution: fem.Function) -> None:
    out = write_snapshot(tmp_path / "bessel.xdmf", bessel_solution, t=0.1)
    h5 = out.with_suffix(".h5")
    assert out.exists() and out.stat().st_size > 0
    # XDMF writes an XML pointer file alongside an HDF5 data file; both
    # must be non-empty for ParaView to load it.
    assert h5.exists() and h5.stat().st_size > 0
    print(f"\n  XDMF: {out.resolve()}\n  HDF5: {h5.resolve()}")


def test_pyvista_screenshot(tmp_path: Path, bessel_solution: fem.Function) -> None:
    png = tmp_path / "bessel.png"
    quick_plot(bessel_solution, screenshot=png, title="J₀ eigenmode")
    assert png.exists()
    # A blank/failed render produces a tiny PNG; a real one with a colormap
    # is comfortably larger. 1 KB is a generous floor.
    assert png.stat().st_size > 1024, f"screenshot only {png.stat().st_size} bytes"
    print(f"\n  PNG:  {png.resolve()}")
