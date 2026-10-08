"""A membrane's maps onto its adjacent compartments (ADR 010 §3, "Membrane adjacency").

A membrane function of the volume concentrations on either side (a flux such as ``P * (c - e)``) is
evaluated at the membrane's points from each compartment's value at the same point. On a body-fitted mesh
the membrane's facets are facets of both compartments' cells, so every membrane point IS a point of each
compartment: the bundle records which one, per segment. The checks:

- every mapped pair coincides exactly, for two compartments and for a nucleus in a cytosol in
  extracellular space (two membranes, three compartments), through the writer and through the runner;
- a field read through the map is the compartment's value at the membrane point;
- each segment carries its own maps (a remesh changes every point index);
- a compartment that is not in the bundle has no map, and a bundle from an older writer has none at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.cli import main
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.results import Bundle, BundleRecorder, BundleWriter, SolverInfo, SourceInfo
from vcell_fenics.results.schema import ADJACENT_VERSION, manifest_from_attrs

_CV = Path(__file__).resolve().parent.parent / "cross_validation"
_SOLVER = SolverInfo(version="test", dolfinx="0.10", mpi_ranks=1)
_SOURCE = SourceInfo(kind="test")
_MEMBRANES: dict[str, tuple[str, ...]] = {"ne": ("nuc", "cyt"), "pm": ("cyt", "ec")}


def _nested() -> GeometryDescription:
    """A nucleus (r = 0.3, off centre) in a cytosol (r = 0.7) in a 2 × 2 box of extracellular space."""

    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="nuc", type="analytic", expression="(geom.x[0] - 0.1) ** 2 + geom.x[1] ** 2 < 0.09"),
            SubVolume(name="cyt", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.49"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="ne", inside="nuc", outside="cyt"),
            SurfaceClass(name="pm", inside="cyt", outside="ec"),
        ),
    )


def _field(scale: float) -> object:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        values: NDArray[np.float64] = scale * (x[0] + 2.0 * x[1] ** 2)
        return values

    return f


def _write(path: Path, geometries: list[MultiCompartmentGeometry], *, volumes: tuple[str, ...]) -> None:
    """Each compartment of ``volumes`` carries u = scale·(x + 2y²) (scale = its index + 1); the membranes
    carry nothing. One row per geometry, each after the first starting a new segment (a remesh)."""

    first = geometries[0]
    times = [float(k) for k in range(len(geometries))]
    writer = BundleWriter(path, comm=MPI.COMM_WORLD, planned_times=times, source=_SOURCE, solver=_SOLVER)
    recorder = BundleRecorder(writer, comm=MPI.COMM_WORLD)
    for k, name in enumerate(volumes):
        recorder.add_domain(name, "volume", first.mesh_of(name), [("u", _field(k + 1.0))])
    for name, membrane in first.membranes.items():
        recorder.add_domain(name, "membrane", membrane.mesh, [], sides=(membrane.inside, membrane.outside))
    recorder.open()
    for row, geometry in enumerate(geometries):
        if row:
            recorder.start_segment(
                {name: (geometry.mesh_of(name), [_field(k + 1.0)]) for k, name in enumerate(volumes)}
                | {name: (geometry.mesh_of(name), []) for name in first.membranes}
            )
        recorder.capture(times[row])
    writer.finalize("completed")


def _check_maps(bundle: Bundle, row: int, membranes: dict[str, tuple[str, ...]]) -> None:
    """Every mapped pair of points coincides exactly; each map is one-to-one."""

    for membrane, compartments in membranes.items():
        points = bundle.mesh(membrane, row).points
        for compartment in compartments:
            mapped = bundle.adjacent(membrane, compartment, row)
            assert mapped is not None, (membrane, compartment)
            assert mapped.dtype == np.int32 and mapped.shape == (points.shape[0],)
            assert (mapped >= 0).all() and np.unique(mapped).size == mapped.size
            assert np.array_equal(bundle.mesh(compartment, row).points[mapped], points), (membrane, compartment)


def test_a_nucleus_in_a_cytosol_maps_both_membranes_onto_both_sides(tmp_path: Path) -> None:
    path = tmp_path / "nested.fenics"
    _write(path, [realize_multi_compartment(_nested(), h=0.15)], volumes=("nuc", "cyt", "ec"))
    bundle = Bundle.open(path)
    for membrane, sides in _MEMBRANES.items():
        adjacency = bundle.manifest.domains[membrane].adjacent
        assert adjacency is not None and adjacency.version == ADJACENT_VERSION
        assert adjacency.compartments == sides
        assert adjacency.maps == {c: f"{membrane}/_adjacent/{c}" for c in sides}
    assert all(bundle.manifest.domains[name].adjacent is None for name in ("nuc", "cyt", "ec"))
    _check_maps(bundle, 0, _MEMBRANES)
    # through the map, a compartment's field at the membrane's points is its value there
    for membrane, sides in _MEMBRANES.items():
        x = bundle.mesh(membrane).points
        for compartment in sides:
            scale = ("nuc", "cyt", "ec").index(compartment) + 1.0
            mapped = bundle.adjacent(membrane, compartment)
            assert mapped is not None
            values = bundle.field(compartment, "u", 0)[mapped]
            assert np.allclose(values, scale * (x[:, 0] + 2.0 * x[:, 1] ** 2), rtol=0, atol=1e-12)


def test_each_segment_has_its_own_maps(tmp_path: Path) -> None:
    path = tmp_path / "segments.fenics"
    coarse, fine = (realize_multi_compartment(_nested(), h=h) for h in (0.2, 0.12))
    _write(path, [coarse, fine], volumes=("nuc", "cyt", "ec"))
    bundle = Bundle.open(path)
    assert [s.prefix for s in bundle.manifest.segments] == ["", "seg0001/"]
    assert (path / "seg0001" / "pm" / "_adjacent" / "ec").is_dir()
    for row in (0, 1):
        _check_maps(bundle, row, _MEMBRANES)
    assert bundle.mesh("pm", 1).points.shape[0] > bundle.mesh("pm", 0).points.shape[0]


def test_a_compartment_outside_the_bundle_has_no_map(tmp_path: Path) -> None:
    path = tmp_path / "partial.fenics"
    _write(path, [realize_multi_compartment(_nested(), h=0.2, compartments={"cyt", "ec"})], volumes=("cyt", "ec"))
    adjacency = Bundle.open(path).manifest.domains["ne"].adjacent
    assert adjacency is not None
    assert adjacency.compartments == ("nuc", "cyt") and set(adjacency.maps) == {"cyt"}
    bundle = Bundle.open(path)
    assert bundle.adjacent("ne", "nuc") is None
    _check_maps(bundle, 0, {"ne": ("cyt",), "pm": ("cyt", "ec")})


def test_a_bundle_from_an_older_writer_has_no_maps(tmp_path: Path) -> None:
    path = tmp_path / "older.fenics"
    _write(path, [realize_multi_compartment(_nested(), h=0.2)], volumes=("nuc", "cyt", "ec"))
    attrs = json.loads((path / ".zattrs").read_text())
    for domain in attrs["vcell_fenics"]["domains"].values():
        domain.pop("adjacent")
    (path / ".zattrs").write_text(json.dumps(attrs))
    manifest = manifest_from_attrs(attrs)
    assert all(d.adjacent is None for d in manifest.domains.values())
    assert Bundle.open(path).adjacent("pm", "cyt") is None


def test_writer_refuses_sides_on_a_volume(tmp_path: Path) -> None:
    geometry = realize_multi_compartment(_nested(), h=0.25)
    path = tmp_path / "b.fenics"
    writer = BundleWriter(path, comm=MPI.COMM_WORLD, planned_times=[0.0], source=_SOURCE, solver=_SOLVER)
    space = fem.functionspace(geometry.mesh_of("cyt"), ("Lagrange", 1))
    with pytest.raises(ValueError, match="only a membrane"):
        writer.add_domain("cyt", "volume", space, sides=("nuc", "ec"))


def test_the_runner_writes_the_maps_of_a_three_compartment_model(tmp_path: Path) -> None:
    # VCell's nucleus model: s_nuc | ne_dom | s_cyto | pm_dom (receptor R) | s_ext. The nuclear envelope has
    # no species of its own, so it is written for its mesh and its maps alone.
    out = tmp_path / "results"
    argv = ["--math", str(_CV / "nucleus_math.yaml"), "--geometry", str(_CV / "nucleus_geom.yaml")]
    assert main([*argv, "--t-final", "0.05", "--output-dt", "0.05", "--h", "0.15", "--out", str(out)]) == 0
    bundle = Bundle.open(out / "results.fenics")
    # compartments first (in equation order), then the membranes: the species-less one last
    assert list(bundle.manifest.domains) == ["nuc_dom", "cyto_dom", "ext_dom", "pm_dom", "ne_dom"]
    written = json.loads((out / "results.fenics" / ".zattrs").read_text())["vcell_fenics"]["domains"]
    assert list(written) == list(bundle.manifest.domains), "the file keeps that order"
    assert not [v for v in bundle.manifest.variables if v.domain == "ne_dom"]
    membranes: dict[str, tuple[str, ...]] = {"ne_dom": ("cyto_dom", "nuc_dom"), "pm_dom": ("cyto_dom", "ext_dom")}
    for membrane, sides in membranes.items():
        adjacency = bundle.manifest.domains[membrane].adjacent
        assert adjacency is not None and set(adjacency.compartments) == set(sides)
    for row in range(len(bundle.times)):
        _check_maps(bundle, row, membranes)
