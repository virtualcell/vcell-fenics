"""The ParaView export (`results/export.py`): every written row, every variable, readable by VTK/meshio.

A bundle stores its fields in zarr beside a field-less VTU, which ParaView cannot join; the export writes
a PVD time series (a VTU per step with point data) or an XDMF3 time series. These tests read each export
back with the libraries ParaView itself builds on and compare it with the bundle.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from dolfinx import mesh as dmesh
from mpi4py import MPI

from vcell_fenics.results import Bundle, BundleRecorder, BundleWriter, SolverInfo, SourceInfo
from vcell_fenics.results.export import export_bundle, main


def _bundle(tmp_path: Path) -> Path:
    square = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    tdim = square.topology.dim
    square.topology.create_connectivity(tdim - 1, tdim)
    edge = dmesh.create_submesh(square, tdim - 1, dmesh.exterior_facet_indices(square.topology))[0]
    path = tmp_path / "b.fenics"
    times = [0.0, 0.5, 1.0]
    writer = BundleWriter(
        path,
        comm=MPI.COMM_WORLD,
        planned_times=times,
        source=SourceInfo(kind="test"),
        solver=SolverInfo(version="t", dolfinx="0.10", mpi_ranks=1),
    )
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    recorder.add_domain("cell", "volume", square, [("u", None), ("v", None)])
    recorder.add_domain("pm", "membrane", edge, [("r", None)])
    recorder.open()
    for t in times:
        recorder.capture(
            t,
            {
                "cell": [lambda x, t=t: (1 + t) * x[0], lambda x, t=t: t * x[1]],
                "pm": [lambda x, t=t: x[0] + x[1] + t],
            },
        )
    writer.finalize("completed")
    return path


def test_pvd_export_carries_every_row_and_variable(tmp_path: Path) -> None:
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    bundle = Bundle.open(_bundle(tmp_path))
    written = export_bundle(bundle, tmp_path / "paraview")
    assert sorted(p.name for p in written) == ["cell.pvd", "pm.pvd"]
    for domain, names in (("cell", ["u", "v"]), ("pm", ["r"])):
        collection = ElementTree.parse(tmp_path / "paraview" / f"{domain}.pvd").getroot()
        entries = collection.findall("./Collection/DataSet")
        assert [float(e.get("timestep", "nan")) for e in entries] == list(bundle.times)
        for row, entry in enumerate(entries):
            reader = vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(tmp_path / "paraview" / entry.get("file", "")))
            reader.Update()
            grid = reader.GetOutput()
            assert grid.GetNumberOfPoints() == bundle.manifest.domains[domain].n_points
            for name in names:
                values = vtk_to_numpy(grid.GetPointData().GetArray(name))
                assert np.array_equal(values, bundle.field(domain, name, row))


def test_xdmf_export_is_a_meshio_time_series(tmp_path: Path) -> None:
    import meshio

    bundle = Bundle.open(_bundle(tmp_path))
    export_bundle(bundle, tmp_path / "xdmf", fmt="xdmf")
    assert sorted(p.name for p in (tmp_path / "xdmf").iterdir()) == ["cell.h5", "cell.xdmf", "pm.h5", "pm.xdmf"]
    assert not Path("cell.h5").exists()  # meshio's working-directory quirk is contained
    with meshio.xdmf.TimeSeriesReader(str(tmp_path / "xdmf" / "cell.xdmf")) as reader:
        points, cells = reader.read_points_cells()
        assert points.shape[0] == bundle.manifest.domains["cell"].n_points
        assert cells[0].type == "triangle"
        for row in range(reader.num_steps):
            t, point_data, _ = reader.read_data(row)
            assert t == pytest.approx(bundle.times[row])
            assert np.allclose(point_data["u"], bundle.field("cell", "u", row))


def test_export_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _bundle(tmp_path)
    assert main([str(path), str(tmp_path / "out")]) == 0
    assert "cell.pvd" in capsys.readouterr().out
    assert main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 2
