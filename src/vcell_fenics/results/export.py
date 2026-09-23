"""Export a results bundle for ParaView (ADR 010 §Decision 3).

ParaView cannot join a bundle's VTU meshes to its zarr fields by itself, so this writes a ParaView-native
copy of every written output row:

- ``pvd`` (default): per domain, one VTU per output time carrying every variable as point data, plus a
  ``<domain>.pvd`` collection that ParaView opens as a time series. Each step is a complete mesh, so this
  also fits a future segmented (moving/remeshed) bundle.
- ``xdmf``: per domain, one XDMF3 + HDF5 time series (meshio's ``TimeSeriesWriter``: the mesh once, the
  fields per step) — the interchange format of the FEniCS world. Fixed-mesh bundles only.

    vcell-fenics-export BUNDLE OUT_DIR [--format pvd|xdmf]      (or python -m vcell_fenics.results.export)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import quoteattr

from vcell_fenics.results.reader import Bundle
from vcell_fenics.results.vtu import VTK_LINE, VTK_TETRA, VTK_TRIANGLE, VtuGrid, write_vtu

ExportFormat = Literal["pvd", "xdmf"]
_MESHIO_CELL = {VTK_LINE: "line", VTK_TRIANGLE: "triangle", VTK_TETRA: "tetra"}


def export_bundle(bundle: Bundle | str | Path, out_dir: Path, *, fmt: ExportFormat = "pvd") -> list[Path]:
    """Write the bundle's written rows in ``fmt`` under ``out_dir``; returns the files ParaView opens
    (one per domain)."""

    source = bundle if isinstance(bundle, Bundle) else Bundle.open(bundle)
    if source.manifest.profile != "fixed" and fmt == "xdmf":
        raise ValueError("XDMF export needs a fixed-mesh bundle; use --format pvd for a segmented one")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _export_domain(source, domain, out_dir, fmt)
        for domain in source.manifest.domains
        if any(v.domain == domain for v in source.manifest.variables)
    ]


def _export_domain(bundle: Bundle, domain: str, out_dir: Path, fmt: ExportFormat) -> Path:
    grid = bundle.mesh(domain)
    variables = [v.name for v in bundle.manifest.variables if v.domain == domain]
    times = bundle.times
    gdim = bundle.manifest.domains[domain].gdim
    vtk_type = int(grid.cell_types[0]) if grid.cell_types.size else VTK_TRIANGLE

    if fmt == "xdmf":
        import meshio

        target = out_dir / f"{domain}.xdmf"
        # meshio's TimeSeriesWriter opens its HDF5 file as "<stem>.h5" relative to the *working
        # directory*, not beside the .xdmf (meshio 5.3.5) — so write from inside out_dir.
        with contextlib.chdir(out_dir), meshio.xdmf.TimeSeriesWriter(target.name) as writer:
            writer.write_points_cells(grid.points[:, :gdim], [(_MESHIO_CELL[vtk_type], grid.cells)])
            series = {name: bundle.series(domain, name) for name in variables}
            for row, t in enumerate(times):
                writer.write_data(t, point_data={name: values[row] for name, values in series.items()})
        return target

    steps = out_dir / domain
    steps.mkdir(exist_ok=True)
    entries: list[str] = []
    meshes: dict[int, VtuGrid] = {}
    for row, t in enumerate(times):
        # a remesh gives each segment its own mesh; an ALE segment moves its points row by row
        segment = bundle.segment_of(row)[0]
        mesh = meshes.setdefault(segment.index, bundle.mesh(domain, row))
        points = bundle.coords(domain, row)[:, : mesh.points.shape[1]] if segment.motion == "ale" else mesh.points
        cell_type = int(mesh.cell_types[0]) if mesh.cell_types.size else vtk_type
        values = {name: bundle.field(domain, name, row) for name in variables}
        step = steps / f"{domain}_{row:05d}.vtu"
        write_vtu(step, points, mesh.cells, cell_type, values)
        entries.append(
            f'    <DataSet timestep={quoteattr(repr(float(t)))} part="0" file={quoteattr(f"{domain}/{step.name}")}/>'
        )
    target = out_dir / f"{domain}.pvd"
    target.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n" + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vcell-fenics-export", description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--format", choices=("pvd", "xdmf"), default="pvd", dest="fmt")
    args = parser.parse_args(argv)
    try:
        written = export_bundle(args.bundle, args.out_dir, fmt=args.fmt)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


__all__ = ["ExportFormat", "export_bundle", "main"]

if __name__ == "__main__":
    sys.exit(main())
