"""The method-of-lines solvers give the serial answer under MPI.

Their default preconditioner, ILU, is sequential in PETSc — under `mpiexec -n 2` every MOL solve failed
("Could not locate a solver type for factorization type ILU and matrix type mpiaij"), found by the first
MPI smoke run of the container image. `backend/linear_solvers.set_preconditioner` substitutes block
Jacobi (ILU(0) per rank) in parallel. This runs a single-mesh MOL solve on a Netgen-realized disk and an
interface-coupled MOL solve, serially and at n=2, and compares totals and extremes. They agree to the
integrator tolerance, not bitwise: the parallel preconditioner (and so every Krylov iterate) differs.
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
from vcell_fenics.backend import assemble, integrate_interface_coupled
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import realize, realize_interface_coupled
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux, MathDescription, ParameterConstant, Subdomain, TemplateEquation, Variable,
)

comm = MPI.COMM_WORLD
out = {}

def total(u):
    mesh = u.function_space.mesh
    return comm.allreduce(float(fem.assemble_scalar(fem.form(u * ufl.dx(domain=mesh))).real), op=MPI.SUM)

def maximum(u):
    n = u.function_space.dofmap.index_map.size_local * u.function_space.dofmap.index_map_bs
    return comm.allreduce(float(u.x.array[:n].max()), op=MPI.MAX)

# (a) single mesh: a nonlinear decay u' = -k u^2 + D Δu on a Netgen disk (so MOL, not backward Euler)
disk = GeometryDescription(
    name="cell", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.49"),
                SubVolume(name="ext", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),))
single = MathDescription(
    geometry="cell",
    subdomains=[Subdomain(name="cyto", kind="volume")],
    variables=[Variable(name="u", subdomain="cyto")],
    parameters=[ParameterConstant(name="k", value=2.0)],
    equations=[TemplateEquation(
        template="bulk_radv_diff", variable="u", subdomain="cyto", temporality="time_dependent",
        terms={"diffusion": "0.1", "source": "-k * u * u"},
        initial_condition="exp(-(pow(geom.x[0] - 0.2, 2) + pow(geom.x[1], 2)) / 0.05)")],
)
problem = assemble(single, realize(disk, h=0.08, comm=comm), dt=0.1)
assert isinstance(problem, DiscreteProblem)
integrate_discrete_problem(problem, t_final=0.5, rtol=1e-8, atol=1e-10)
out["single_total"], out["single_max"] = total(problem.unknown), maximum(problem.unknown)

# (b) two compartments coupled by a permeability flux across their membrane
nested = GeometryDescription(
    name="nested", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="inner", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.16"),
                SubVolume(name="outer", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.64"),
                SubVolume(name="bg", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="m", inside="inner", outside="outer"),
              SurfaceClass(name="w", inside="outer", outside="bg")))
coupled = MathDescription(
    geometry="nested",
    subdomains=[Subdomain(name="inner", kind="volume"), Subdomain(name="outer", kind="volume")],
    variables=[Variable(name="a", subdomain="inner"), Variable(name="b", subdomain="outer")],
    parameters=[ParameterConstant(name="P", value=2.0)],
    equations=[
        TemplateEquation(template="bulk_radv_diff", variable="a", subdomain="inner",
                         temporality="time_dependent", terms={"diffusion": "1.0"}, initial_condition="1.0"),
        TemplateEquation(template="bulk_radv_diff", variable="b", subdomain="outer",
                         temporality="time_dependent", terms={"diffusion": "1.0"}, initial_condition="0.0"),
    ],
    boundary_conditions=[
        BCInterfaceFlux(variable="a", boundary="pm", expression="P * (b - a)"),
        BCInterfaceFlux(variable="b", boundary="pm", expression="P * (a - b)"),
    ],
)
geometry = realize_interface_coupled(nested, inner_subdomain="inner", outer_subdomain="outer",
                                     membrane_subdomain="m", interface="pm", background_subdomain="bg",
                                     h=0.1, comm=comm)
result = integrate_interface_coupled(coupled, geometry, t_final=0.5, rtol=1e-8, atol=1e-10)
out["coupled_inner"], out["coupled_outer"] = total(result.inner), total(result.outer)
if comm.rank == 0:
    print("PROBE " + json.dumps(out))
"""


def _run(n: int) -> dict[str, float]:
    mpiexec = Path(sys.executable).parent / "mpiexec"
    result = subprocess.run(
        [str(mpiexec), "-n", str(n), sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, f"n={n} failed:\n{result.stderr[-4000:]}"
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PROBE "))
    parsed: dict[str, float] = json.loads(line.removeprefix("PROBE "))
    return parsed


@pytest.mark.integration
def test_method_of_lines_solves_agree_under_mpi() -> None:
    serial, parallel = _run(1), _run(2)
    assert parallel.keys() == serial.keys()
    for key, value in serial.items():
        assert parallel[key] == pytest.approx(value, rel=1e-5), key
    # the coupled pair conserves its substance whatever the rank count
    assert parallel["coupled_inner"] + parallel["coupled_outer"] == pytest.approx(
        serial["coupled_inner"] + serial["coupled_outer"], rel=1e-9
    )
