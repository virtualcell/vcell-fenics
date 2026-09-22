"""`realize` / `realize_interface_coupled` give the same geometry under MPI as serially.

The body-fitted paths mesh with Netgen, which is serial. Two parallel bugs lived here (found
2026-09-22 while wiring vcell-fenics into VCell, `docs/integration/vcell-solver-integration.md`):

- every rank handed its *own* copy of the Netgen mesh to `create_mesh`, which reads each rank's
  arrays as its share of one global input, so under `mpiexec -n 2` the partition path indexed its
  material tags out of range and crashed;
- membrane facets were found by "two incident cells", which without ghost cells misses any membrane
  facet on a partition boundary (a 3D sphere lost one of 120 membrane facets at n=2).

The fix meshes on rank 0 only and ghosts across shared facets. This test runs the same realizations
serially and at n=2 and n=3 and compares global cell/node counts and measures (areas, volumes, membrane
lengths/areas, interface-coupled wall and interface facet counts) — they must agree exactly, because the
mesh itself is identical; only its distribution differs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = r"""
import json
import ufl
from dolfinx import fem
from mpi4py import MPI
from vcell_fenics.backend.realize import realize, realize_interface_coupled
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

comm = MPI.COMM_WORLD
out = {}

def measure(msh):
    return comm.allreduce(float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh))).real), op=MPI.SUM)

def record(key, msh):
    tdim = msh.topology.dim
    out[key] = [msh.topology.index_map(tdim).size_global, msh.geometry.index_map().size_global, round(measure(msh), 12)]

disk = GeometryDescription(
    name="disk", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.49"),
                SubVolume(name="ext", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),))
sphere = GeometryDescription(
    name="sphere", dim=3, extent=(2.0, 2.0, 2.0), origin=(-1.0, -1.0, -1.0),
    subvolumes=(SubVolume(name="cyto", type="analytic",
                          expression="geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < 0.36"),
                SubVolume(name="ext", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),))
for desc, h in ((disk, 0.1), (sphere, 0.3)):
    geom = realize(desc, h=h, comm=comm)
    for sd in sorted(geom.subdomains):
        record(f"{desc.name}/{sd}", geom.mesh_of(sd))
    for face in sorted(geom.boundaries):
        facets = geom.boundaries[face].facets
        parent = geom.parent_mesh
        owned = facets[facets < parent.topology.index_map(parent.topology.dim - 1).size_local]
        out[f"{desc.name}/boundary/{face}"] = [comm.allreduce(int(owned.size), op=MPI.SUM),
                                               list(geom.boundaries[face].subdomains)]

nested = GeometryDescription(
    name="nested", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="inner", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.16"),
                SubVolume(name="outer", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.64"),
                SubVolume(name="bg", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="m", inside="inner", outside="outer"),
              SurfaceClass(name="w", inside="outer", outside="bg")))
for bg in (None, "bg"):
    g = realize_interface_coupled(nested, inner_subdomain="inner", outer_subdomain="outer",
                                  membrane_subdomain="m", interface="pm", background_subdomain=bg,
                                  h=0.1, comm=comm)
    for name, msh in (("inner", g.inner_mesh), ("outer", g.outer_mesh), ("membrane", g.membrane_mesh)):
        record(f"coupled[{bg}]/{name}", msh)
    n_owned = g.parent_mesh.topology.index_map(g.parent_mesh.topology.dim - 1).size_local
    for label, tag in (("wall", g.outer_tag), ("interface", g.interface_tag)):
        facets = g.facet_tags.find(tag)
        out[f"coupled[{bg}]/{label}_facets"] = comm.allreduce(int((facets < n_owned).sum()), op=MPI.SUM)

if comm.rank == 0:
    print("PROBE " + json.dumps(out, sort_keys=True))
"""


def _run(n: int) -> dict[str, object]:
    mpiexec = Path(sys.executable).parent / "mpiexec"
    result = subprocess.run(
        [str(mpiexec), "-n", str(n), sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"n={n} failed:\n{result.stderr[-4000:]}"
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PROBE "))
    parsed: dict[str, object] = json.loads(line.removeprefix("PROBE "))
    return parsed


@pytest.mark.integration
@pytest.mark.parametrize("n", [2, 3])
def test_realized_geometry_is_independent_of_rank_count(n: int) -> None:
    serial = _run(1)
    parallel = _run(n)
    assert parallel.keys() == serial.keys()
    mismatched = {key: (serial[key], parallel[key]) for key in serial if serial[key] != parallel[key]}
    assert not mismatched, f"n={n} differs from serial: {mismatched}"
