"""Visualization helpers — PyVista for in-process, XDMF for ParaView.

Both entry points take any ``dolfinx.fem.Function`` (or a list of them), so
solutions, initial conditions, and residuals (also Functions) flow through
the same code paths. Residuals are not computed here; the caller builds a
Function from whatever residual expression they want.

Long-term these will likely live under ``core/io.py`` and ``core/viz.py``
per the architecture sketch in ``docs/modeling/approaches.md``; placed at
the package root for now to avoid creating ``core/`` for a single file.

PyVista is a dev-environment dependency only (the runtime image does not ship it), so it is
imported inside the functions that render; the XDMF writers work in either environment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from dolfinx import fem, plot
from dolfinx.io import XDMFFile
from mpi4py import MPI

if TYPE_CHECKING:
    import pyvista


def _function_to_pyvista(field: fem.Function) -> pyvista.UnstructuredGrid:
    """Wrap a Function's mesh and DOF values in a PyVista UnstructuredGrid.

    Uses ``dolfinx.plot.vtk_mesh`` so higher-order elements are linearized
    in a viz-only sense; values at the linear vertices are correct.
    """
    import pyvista

    # vtk_mesh accepts a FunctionSpace at runtime; the stub only types Mesh.
    topology, cell_types, geometry = plot.vtk_mesh(field.function_space)  # type: ignore[arg-type]
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[field.name or "field"] = field.x.array.real
    grid.set_active_scalars(field.name or "field")
    return grid


def quick_plot(
    field: fem.Function,
    *,
    screenshot: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
    show_edges: bool = True,
    scalar_bar: bool = True,
    window_size: tuple[int, int] = (800, 600),
    cmap: str = "viridis",
) -> pyvista.Plotter:
    """Render a scalar Function via PyVista.

    If ``screenshot`` is given, save a PNG to that path (off-screen render).
    If ``show`` is True, open an interactive window. Both may be combined.
    Returns the Plotter so callers can compose additional actors before
    re-rendering if needed.
    """
    import pyvista

    grid = _function_to_pyvista(field)
    off_screen = bool(screenshot) and not show
    plotter = pyvista.Plotter(off_screen=off_screen, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        scalars=field.name or "field",
        show_edges=show_edges,
        cmap=cmap,  # type: ignore[arg-type]  # pyvista's stub types cmap as a huge Literal; any str is valid
        show_scalar_bar=scalar_bar,
    )
    if title:
        plotter.add_text(title, font_size=10)
    plotter.view_xy()  # type: ignore[call-arg]  # pyvista stub bug: view_xy wrapped as a method missing self

    if screenshot is not None:
        plotter.screenshot(str(screenshot))
    if show:
        plotter.show()
    if not show:
        plotter.close()
    return plotter


def write_snapshot(
    path: str | Path,
    fields: fem.Function | Sequence[fem.Function],
    *,
    t: float = 0.0,
    comm: MPI.Comm | None = None,
) -> Path:
    """Write fields to an XDMF file at time ``t``.

    All fields must share a mesh. XDMF is chosen as the default because
    ParaView reads it without any plugins; use ``write_series`` (TBD) for
    long time series where ADIOS2/VTKHDF would scale better.
    """
    out_path = Path(path)
    field_list: list[fem.Function] = [fields] if isinstance(fields, fem.Function) else list(fields)
    if not field_list:
        raise ValueError("write_snapshot requires at least one field")

    mesh = field_list[0].function_space.mesh
    if comm is None:
        comm = mesh.comm

    with XDMFFile(comm, str(out_path), "w") as xf:
        xf.write_mesh(mesh)
        for f in field_list:
            xf.write_function(f, t)
    return out_path


def write_series(
    path: str | Path,
    fields: Sequence[fem.Function],
    times: Iterable[float],
    step_fn: Callable[[float], None],
    *,
    comm: MPI.Comm | None = None,
) -> Path:
    """Drive a time loop and append fields to XDMF on each step.

    ``step_fn`` is called once per time in ``times``; the same Functions
    are re-written with updated DOF values. Use this for actual runs;
    for one-off snapshots prefer ``write_snapshot``.
    """
    out_path = Path(path)
    field_list = list(fields)
    if not field_list:
        raise ValueError("write_series requires at least one field")

    mesh = field_list[0].function_space.mesh
    if comm is None:
        comm = mesh.comm

    with XDMFFile(comm, str(out_path), "w") as xf:
        xf.write_mesh(mesh)
        for t in times:
            step_fn(t)
            for f in field_list:
                xf.write_function(f, t)
    return out_path
