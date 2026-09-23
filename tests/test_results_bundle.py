"""The results bundle (ADR 010): VTU meshes VCell can parse, zarr fields whose columns follow the VTU
point order, per-time statistics, and a manifest that makes rows visible only once they are complete.

The fields written here are analytic (a P1 interpolant equals the function at its nodes), so a value
read back can be checked against the coordinates of the point it claims to belong to — the test that
matters for a viewer that pairs `mesh/<domain>.vtu` points with `<domain>/<var>` columns.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from dolfinx import fem
from dolfinx.mesh import CellType, create_box, create_unit_square
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.results import Bundle, BundleRecorder, BundleSchemaError, BundleWriter, SolverInfo, SourceInfo
from vcell_fenics.results import reader as reader_cli
from vcell_fenics.results.schema import dumps_json_schema, manifest_from_attrs
from vcell_fenics.results.vtu import VTK_LINE, VTK_TETRA, VTK_TRIANGLE, read_vtu_strict, write_vtu

_SCHEMA_DOC = Path(__file__).resolve().parent.parent / "docs" / "results-bundle.schema.json"
_SOLVER = SolverInfo(version="test", dolfinx="0.10", mpi_ranks=1)
_SOURCE = SourceInfo(kind="test")


def _u(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values: NDArray[np.float64] = x[0] + 2.0 * x[1]
    return values


def _r(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values: NDArray[np.float64] = x[0] * x[1] + 1.0
    return values


def _disk() -> GeometryDescription:
    return GeometryDescription(
        name="disk",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.49"),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )


def _write_disk_bundle(path: Path, times: list[float], *, finalize: bool = True) -> BundleRecorder:
    geometry = realize(_disk(), h=0.15)
    writer = BundleWriter(path, comm=MPI.COMM_WORLD, planned_times=times, source=_SOURCE, solver=_SOLVER)
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    recorder.add_domain("cyto", "volume", geometry.mesh_of("cyto"), [("u", _u)])
    recorder.add_domain("pm", "membrane", geometry.mesh_of("pm"), [("r", _r)])
    recorder.open()
    for t in times:
        scale = 1.0 + t
        recorder.capture(
            t,
            {"cyto": [lambda x, s=scale: s * _u(x)], "pm": [lambda x, s=scale: s * _r(x)]},
            progress=t / times[-1] if times[-1] else 1.0,
        )
    if finalize:
        writer.finalize("completed")
    return recorder


# -- VTU ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "points,cells,vtk_type",
    [
        (np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]]), np.array([[0, 1, 2], [1, 3, 2]]), VTK_TRIANGLE),
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]), np.array([[0, 1, 2, 3]]), VTK_TETRA),
        (np.array([[0, 0], [1, 0], [1, 1.0]]), np.array([[0, 1], [1, 2]]), VTK_LINE),
    ],
)
def test_vtu_round_trips_through_the_vtugridparser_mirror(
    tmp_path: Path, points: NDArray[np.float64], cells: NDArray[np.int64], vtk_type: int
) -> None:
    path = tmp_path / "mesh.vtu"
    write_vtu(path, points, cells, vtk_type)
    grid = read_vtu_strict(path)
    assert np.allclose(grid.points[:, : points.shape[1]], points)
    assert np.array_equal(grid.cells, cells)
    assert np.all(grid.cell_types == vtk_type)
    header = path.read_bytes()[:300].decode()
    assert 'header_type="UInt32"' in header and "compressor" not in header


def test_strict_reader_refuses_compressed_vtu(tmp_path: Path) -> None:
    path = tmp_path / "mesh.vtu"
    write_vtu(path, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]), VTK_TRIANGLE)
    path.write_text(path.read_text().replace('byte_order="LittleEndian"', 'byte_order="LittleEndian" compressor="x"'))
    with pytest.raises(ValueError, match="misread"):
        read_vtu_strict(path)


# -- fields and meshes -----------------------------------------------------------------------------------


def test_field_columns_follow_the_vtu_point_order(tmp_path: Path) -> None:
    times = [0.0, 0.5, 1.0]
    _write_disk_bundle(tmp_path / "b.fenics", times)
    bundle = Bundle.open(tmp_path / "b.fenics")

    assert bundle.status == "completed"
    assert bundle.times == tuple(times)
    for domain, variable, exact, vtk_type in (("cyto", "u", _u, VTK_TRIANGLE), ("pm", "r", _r, VTK_LINE)):
        grid = bundle.mesh(domain)
        assert np.all(grid.cell_types == vtk_type)
        assert bundle.manifest.domains[domain].n_points == grid.points.shape[0]
        series = bundle.series(domain, variable)
        assert series.shape == (len(times), grid.points.shape[0])
        for row, t in enumerate(times):
            assert np.allclose(series[row], (1.0 + t) * exact(grid.points.T), rtol=1e-12, atol=1e-12)


def test_statistics_are_the_reduced_integrals(tmp_path: Path) -> None:
    recorder = _write_disk_bundle(tmp_path / "b.fenics", [0.0, 1.0])
    bundle = Bundle.open(tmp_path / "b.fenics")
    stats = bundle.stats("cyto", "u")  # columns: mean, total, min, max
    area = recorder.measure("cyto")
    assert area == pytest.approx(np.pi * 0.49, rel=5e-2)
    assert np.allclose(stats[:, 0], stats[:, 1] / area)
    # x + 2y is odd over a region symmetric about the origin: its integral is ≈ 0 up to the faceting.
    assert np.all(np.abs(stats[:, 1]) < 1e-2)
    grid, values = bundle.mesh("cyto"), bundle.series("cyto", "u")
    assert np.allclose(stats[:, 2], values.min(axis=1)) and np.allclose(stats[:, 3], values.max(axis=1))
    assert bundle.manifest.stats_columns == ("mean", "total", "min", "max")
    assert grid.points.shape[0] == values.shape[1]


def test_cells_are_positively_oriented(tmp_path: Path) -> None:
    _write_disk_bundle(tmp_path / "b.fenics", [0.0])
    grid = Bundle.open(tmp_path / "b.fenics").mesh("cyto")
    p = grid.points[:, :2]
    a, b, c = p[grid.cells[:, 0]], p[grid.cells[:, 1]], p[grid.cells[:, 2]]
    signed = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert np.all(signed > 0.0)


def test_tetrahedral_domain(tmp_path: Path) -> None:
    mesh = create_box(MPI.COMM_WORLD, [np.zeros(3), np.ones(3)], [2, 2, 2], CellType.tetrahedron)
    writer = BundleWriter(
        tmp_path / "t.fenics", comm=MPI.COMM_WORLD, planned_times=[0.0], source=_SOURCE, solver=_SOLVER
    )
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    recorder.add_domain("cell", "volume", mesh, [("c", lambda x: x[0] + x[1] * x[2])])
    recorder.open()
    recorder.capture(0.0)
    writer.finalize("completed")
    bundle = Bundle.open(tmp_path / "t.fenics")
    grid = bundle.mesh("cell")
    assert np.all(grid.cell_types == VTK_TETRA) and grid.cells.shape == (48, 4)
    assert np.allclose(bundle.field("cell", "c", 0), grid.points[:, 0] + grid.points[:, 1] * grid.points[:, 2])
    p = grid.points
    edges = np.stack([p[grid.cells[:, k]] - p[grid.cells[:, 0]] for k in (1, 2, 3)], axis=1)
    assert np.all(np.linalg.det(edges) > 0.0)


# -- the manifest, mid-run reads, and the raw encoding ---------------------------------------------------


def test_rows_become_visible_only_through_the_manifest(tmp_path: Path) -> None:
    path = tmp_path / "b.fenics"
    _write_disk_bundle(path, [0.0, 0.5, 1.0, 1.5], finalize=False)
    # Simulate a reader that arrives mid-run: trim the manifest back to two rows.
    attrs = json.loads((path / ".zattrs").read_text())
    attrs["vcell_fenics"]["times"] = attrs["vcell_fenics"]["times"][:2]
    (path / ".zattrs").write_text(json.dumps(attrs))
    bundle = Bundle.open(path)
    assert bundle.status == "running"
    assert bundle.series("cyto", "u").shape[0] == 2
    with pytest.raises(IndexError, match="not written"):
        bundle.field("cyto", "u", 2)


def test_a_bundle_appears_complete_or_not_at_all(tmp_path: Path) -> None:
    """open() builds in a staging directory and renames it into place: no half-made bundle is ever
    visible, and a previous bundle at the path is replaced."""

    path = tmp_path / "b.fenics"
    _write_disk_bundle(path, [0.0, 1.0])
    first = Bundle.open(path).manifest.updated
    _write_disk_bundle(path, [0.0])  # a re-run replaces it
    assert Bundle.open(path).times == (0.0,) and Bundle.open(path).manifest.updated >= first
    assert [p.name for p in tmp_path.iterdir()] == ["b.fenics"]  # no staging directory left behind


def test_preallocated_rows_read_as_nan(tmp_path: Path) -> None:
    geometry = realize(_disk(), h=0.3)
    path = tmp_path / "b.fenics"
    writer = BundleWriter(path, comm=MPI.COMM_WORLD, planned_times=[0.0, 1.0, 2.0], source=_SOURCE, solver=_SOLVER)
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    recorder.add_domain("cyto", "volume", geometry.mesh_of("cyto"), [("u", _u)])
    recorder.open()
    recorder.capture(0.0)
    writer.finalize("failed", "stopped early")

    import zarr

    raw = zarr.open_group(str(path), mode="r", zarr_format=2)["cyto/u"]
    assert isinstance(raw, zarr.Array)
    assert raw.shape[0] == 3 and np.isnan(np.asarray(raw[2, :]))[0]
    bundle = Bundle.open(path)
    assert bundle.status == "failed" and bundle.manifest.message == "stopped early"
    assert bundle.series("cyto", "u").shape[0] == 1


def test_a_row_decodes_with_the_standard_library(tmp_path: Path) -> None:
    """The 'tiny Java/JS reader' claim: .zarray JSON + zlib + little-endian doubles, nothing else."""

    path = tmp_path / "b.fenics"
    _write_disk_bundle(path, [0.0, 1.0])
    meta = json.loads((path / "cyto" / "u" / ".zarray").read_text())
    assert meta["zarr_format"] == 2 and meta["compressor"]["id"] == "zlib" and meta["dtype"] == "<f8"
    row = np.frombuffer(zlib.decompress((path / "cyto" / "u" / "1.0").read_bytes()), dtype="<f8")
    assert np.array_equal(row, Bundle.open(path).field("cyto", "u", 1))
    assert json.loads((path / "cyto" / "u" / ".zattrs").read_text())["_ARRAY_DIMENSIONS"] == ["time", "point"]


def test_reader_ignores_unknown_keys_and_refuses_a_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "b.fenics"
    _write_disk_bundle(path, [0.0])
    attrs = json.loads((path / ".zattrs").read_text())
    attrs["vcell_fenics"]["added_by_a_newer_writer"] = {"x": 1}
    attrs["vcell_fenics"]["domains"]["cyto"]["also_new"] = True
    assert manifest_from_attrs(attrs).domains["cyto"].n_points > 0
    attrs["vcell_fenics"]["schema"] = 2
    with pytest.raises(BundleSchemaError, match="not supported"):
        manifest_from_attrs(attrs)


def test_published_json_schema_is_current() -> None:
    """`docs/results-bundle.schema.json` is the contract Java/JS readers see; regenerate it with
    `python -c "from vcell_fenics.results.schema import dumps_json_schema; print(dumps_json_schema(), end='')"`."""

    assert _SCHEMA_DOC.read_text() == dumps_json_schema()


def test_reader_cli_checks_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "b.fenics"
    _write_disk_bundle(path, [0.0, 1.0])
    assert reader_cli.main([str(path), "--require-status", "completed"]) == 0
    assert "2/2 rows" in capsys.readouterr().out
    assert reader_cli.main([str(path), "--require-status", "running"]) == 1
    assert reader_cli.main([str(tmp_path / "missing")]) == 2


def test_writer_rejects_an_incomplete_row(tmp_path: Path) -> None:
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    writer = BundleWriter(
        tmp_path / "b.fenics", comm=MPI.COMM_WORLD, planned_times=[0.0], source=_SOURCE, solver=_SOLVER
    )
    space = fem.functionspace(mesh, ("Lagrange", 1))
    writer.add_domain("d", "volume", space)
    writer.add_variable("d", "a")
    writer.add_variable("d", "b")
    writer.open()
    with pytest.raises(ValueError, match="every registered"):
        writer.write(0.0, {("d", "a"): np.zeros(9)}, {("d", "a"): (0.0, 0.0, 0.0, 0.0)})


# -- moving meshes: ALE coordinates per row, a new segment per remesh (ADR 010 §2–3) ------------------


def _write_moving_bundle(path: Path, times: list[float], *, remesh_after: int | None = None) -> None:
    """A disk that translates by (t, 0) and dilates by (1 + t) between rows, carrying the field
    u = x + 2y evaluated on the *moved* mesh — so a row's values must pair with that row's coordinates.
    With ``remesh_after``, the domain is re-meshed (a finer disk, a different point count) after that row."""

    geometry = realize(_disk(), h=0.2)
    mesh = geometry.mesh_of("cyto")
    base: NDArray[np.float64] = np.array(mesh.geometry.x, dtype=np.float64, copy=True)
    writer = BundleWriter(path, comm=MPI.COMM_WORLD, planned_times=times, source=_SOURCE, solver=_SOLVER)
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    recorder.add_domain("cyto", "volume", mesh, [("u", _u)], moving=True)
    recorder.open()
    for row, t in enumerate(times):
        mesh.geometry.x[:, :2] = base[:, :2] * (1.0 + t) + np.array([t, 0.0])
        recorder.capture(t)
        if remesh_after is not None and row == remesh_after:
            finer = realize(_disk(), h=0.12).mesh_of("cyto")
            base = np.array(finer.geometry.x, dtype=np.float64, copy=True)
            mesh = finer
            recorder.start_segment({"cyto": (mesh, [_u])})
    writer.finalize("completed")


def test_a_moving_domain_records_its_coordinates_every_row(tmp_path: Path) -> None:
    path = tmp_path / "moving.fenics"
    times = [0.0, 0.25, 0.5]
    _write_moving_bundle(path, times)
    bundle = Bundle.open(path)
    assert bundle.manifest.profile == "segmented"
    (segment,) = bundle.manifest.segments
    assert (segment.motion, segment.prefix, segment.count) == ("ale", "", 3)
    reference = bundle.coords("cyto", 0)
    for row, t in enumerate(times):
        coords = bundle.coords("cyto", row)
        # the recorded points are the moved ones, and the field's columns belong to them
        assert np.allclose(coords[:, :2], (reference[:, :2]) * (1.0 + t) + [t, 0.0], atol=1e-12)
        assert np.allclose(bundle.field("cyto", "u", row), coords[:, 0] + 2.0 * coords[:, 1], atol=1e-12)
    # the measure follows the dilation: mean = total / |domain(t)|, and |domain| grows as (1 + t)^2
    area0 = bundle.stats("cyto", "u")[0, 1] / bundle.stats("cyto", "u")[0, 0]
    stats = bundle.stats("cyto", "u")
    for row, t in enumerate(times):
        assert stats[row, 1] / stats[row, 0] == pytest.approx(area0 * (1.0 + t) ** 2, rel=1e-10)


def test_a_remesh_starts_a_segment_with_its_own_mesh(tmp_path: Path) -> None:
    path = tmp_path / "remeshed.fenics"
    times = [0.0, 0.25, 0.5, 0.75]
    _write_moving_bundle(path, times, remesh_after=1)
    bundle = Bundle.open(path)
    first, second = bundle.manifest.segments
    assert (first.count, second.count) == (2, 2)
    assert (second.prefix, second.motion, second.t0) == ("seg0001/", "ale", 0.5)
    assert (path / "seg0001" / "mesh" / "cyto.vtu").is_file()
    n0 = bundle.mesh("cyto", 0).points.shape[0]
    n1 = bundle.mesh("cyto", 2).points.shape[0]
    assert n1 > n0  # the finer remesh
    for row in range(len(times)):
        coords = bundle.coords("cyto", row)
        values = bundle.field("cyto", "u", row)
        assert values.shape[0] == coords.shape[0] == (n0 if row < 2 else n1)
        assert np.allclose(values, coords[:, 0] + 2.0 * coords[:, 1], atol=1e-12)
    assert bundle.stats("cyto", "u").shape == (4, 4)
    with pytest.raises(ValueError, match="remeshed"):
        bundle.series("cyto", "u")


def test_pvd_export_follows_the_moving_mesh(tmp_path: Path) -> None:
    from vcell_fenics.results.export import export_bundle

    path = tmp_path / "remeshed.fenics"
    _write_moving_bundle(path, [0.0, 0.25, 0.5, 0.75], remesh_after=1)
    (pvd,) = export_bundle(path, tmp_path / "paraview")
    assert pvd.name == "cyto.pvd"
    bundle = Bundle.open(path)
    for row in range(4):
        step = read_vtu_strict(tmp_path / "paraview" / "cyto" / f"cyto_{row:05d}.vtu")
        assert np.allclose(step.points, bundle.coords("cyto", row), atol=1e-12)
    with pytest.raises(ValueError, match="XDMF"):
        export_bundle(path, tmp_path / "xdmf", fmt="xdmf")


def test_reserved_bundle_names(tmp_path: Path) -> None:
    writer = BundleWriter(
        tmp_path / "b.fenics", comm=MPI.COMM_WORLD, planned_times=[0.0], source=_SOURCE, solver=_SOLVER
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    writer.add_domain("cyto", "volume", fem.functionspace(mesh, ("Lagrange", 1)))
    for bad in ("_coords", "seg0001"):
        with pytest.raises(ValueError, match="cannot name"):
            writer.add_variable("cyto", bad)
