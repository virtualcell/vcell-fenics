"""A results bundle written under MPI is the one written serially: same VTU points and cells, same field
columns — the canonical P1 layout (`results/gather.py`, ADR 010 §6(f)) at work end to end.

Runs the same realization + write at n=1, 2 and 3 in subprocesses (a disk with its membrane, so a
volume *and* a surface domain) and compares the bundles, the membrane's maps onto its sides included.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vcell_fenics.results import Bundle

_WRITE = r"""
import sys
from mpi4py import MPI
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.results import BundleRecorder, BundleWriter, SolverInfo, SourceInfo

comm = MPI.COMM_WORLD
disk = GeometryDescription(
    name="disk", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.49"),
                SubVolume(name="ext", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),))
geometry = realize(disk, h=0.12, comm=comm)
times = [0.0, 0.5, 1.0]
writer = BundleWriter(sys.argv[1], comm=comm, planned_times=times, source=SourceInfo(kind="test"),
                      solver=SolverInfo(version="test", dolfinx="0.10", mpi_ranks=comm.size))
recorder = BundleRecorder(writer, comm=comm)
recorder.add_domain("cyto", "volume", geometry.mesh_of("cyto"), [("u", lambda x: x[0] + 2 * x[1])])
recorder.add_domain("ext", "volume", geometry.mesh_of("ext"), [("v", lambda x: x[0] * x[1])])
recorder.add_domain("pm", "membrane", geometry.mesh_of("pm"), [("r", lambda x: x[0] - x[1] ** 2)],
                    sides=("cyto", "ext"))
recorder.open()
for t in times:
    s = 1.0 + t
    recorder.capture(t, {"cyto": [lambda x, s=s: s * (x[0] + 2 * x[1])],
                         "ext": [lambda x, s=s: s * x[0] * x[1]],
                         "pm": [lambda x, s=s: s * (x[0] - x[1] ** 2)]})
writer.finalize("completed")
"""


def _write(n: int, path: Path) -> Bundle:
    mpiexec = Path(sys.executable).parent / "mpiexec"
    result = subprocess.run(
        [str(mpiexec), "-n", str(n), sys.executable, "-c", _WRITE, str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"n={n} failed:\n{result.stderr[-4000:]}"
    return Bundle.open(path)


@pytest.mark.integration
@pytest.mark.parametrize("n", [2, 3])
def test_bundle_is_independent_of_rank_count(tmp_path: Path, n: int) -> None:
    serial = _write(1, tmp_path / "serial.fenics")
    parallel = _write(n, tmp_path / f"n{n}.fenics")
    assert parallel.manifest.solver.mpi_ranks == n
    assert parallel.times == serial.times
    for variable in serial.manifest.variables:
        domain = variable.domain
        a, b = serial.mesh(domain), parallel.mesh(domain)
        assert np.array_equal(a.points, b.points), domain
        assert np.array_equal(a.cells, b.cells), domain
        assert np.allclose(
            serial.series(domain, variable.name), parallel.series(domain, variable.name), rtol=0, atol=1e-13
        )
        assert np.allclose(
            serial.stats(domain, variable.name), parallel.stats(domain, variable.name), rtol=1e-12, atol=1e-13
        )
    # The membrane's maps onto its compartments are the serial ones, and pair coinciding points.
    for compartment in ("cyto", "ext"):
        serial_map, parallel_map = serial.adjacent("pm", compartment), parallel.adjacent("pm", compartment)
        assert serial_map is not None and parallel_map is not None
        assert np.array_equal(serial_map, parallel_map), compartment
        assert np.array_equal(parallel.mesh(compartment).points[parallel_map], parallel.mesh("pm").points)
    # The VTU is byte-identical, not merely equal after parsing.
    for name, domain_info in serial.manifest.domains.items():
        assert (serial.path / domain_info.mesh).read_bytes() == (parallel.path / domain_info.mesh).read_bytes(), name
