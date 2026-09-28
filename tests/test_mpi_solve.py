"""The method-of-lines solvers give the serial answer under MPI.

Their default preconditioner, ILU, is sequential in PETSc — under `mpiexec -n 2` every MOL solve failed
("Could not locate a solver type for factorization type ILU and matrix type mpiaij"), found by the first
MPI smoke run of the container image. `backend/linear_solvers.set_preconditioner` substitutes block
Jacobi (ILU(0) per rank) in parallel. This runs a single-mesh MOL solve on a Netgen-realized disk and an
interface-coupled MOL solve (with and without a region variable, a Real block), serially and at n=2, and
compares totals and extremes. They agree to the
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

# (c) the same pair with a membrane potential (a region variable: one Real DOF, owned by one rank) gating the
# permeability, C dV/dt = -g (V - E): the Real block must give the serial answer too
from dataclasses import replace
gated = replace(
    coupled,
    subdomains=[*coupled.subdomains, Subdomain(name="m", kind="surface")],
    variables=[*coupled.variables, Variable(name="V", subdomain="m", space="region")],
    parameters=[*coupled.parameters, ParameterConstant(name="g", value=2.0), ParameterConstant(name="E", value=-60.0)],
    equations=[*coupled.equations, TemplateEquation(template="region_ode", variable="V", subdomain="m",
               temporality="time_dependent", terms={"region_rate": "-g * (V - E)"}, initial_condition="-20.0")],
    boundary_conditions=[
        BCInterfaceFlux(variable="a", boundary="pm", expression="P * (1.0 + 0.01 * (V + 60.0)) * (b - a)"),
        BCInterfaceFlux(variable="b", boundary="pm", expression="P * (1.0 + 0.01 * (V + 60.0)) * (a - b)"),
    ],
)
region = integrate_interface_coupled(gated, geometry, t_final=0.5, rtol=1e-8, atol=1e-10)
potential = region.fields["V"]
n_owned = potential.function_space.dofmap.index_map.size_local
out["region_V"] = comm.allreduce(float(potential.x.array[:n_owned].sum()), op=MPI.SUM)
out["region_inner"], out["region_outer"] = total(region.inner), total(region.outer)

# (d) the multi-compartment solver: a nucleus in a cytosol in extracellular space, a receptor on the plasma
# membrane binding the outside ligand (matrix-free Newton, membrane-mesh and parent forms, three submeshes)
from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment, species_mass
three = GeometryDescription(
    name="three", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0),
    subvolumes=(SubVolume(name="nuc", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.09"),
                SubVolume(name="cyt", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.36"),
                SubVolume(name="ec", type="analytic", expression="1.0")),
    surfaces=(SurfaceClass(name="ne", inside="nuc", outside="cyt"),
              SurfaceClass(name="pm", inside="cyt", outside="ec")))
def pde(v, sd, ic, template="bulk_radv_diff", **terms):
    return TemplateEquation(template=template, variable=v, subdomain=sd, temporality="time_dependent",
                            terms={"diffusion": "1.0", **terms}, initial_condition=ic)
multi = MathDescription(
    geometry="three",
    subdomains=[Subdomain(name=n, kind="volume") for n in ("nuc", "cyt", "ec")]
    + [Subdomain(name=n, kind="surface") for n in ("ne", "pm")],
    variables=[Variable(name="n", subdomain="nuc"), Variable(name="c", subdomain="cyt"),
               Variable(name="e", subdomain="ec"), Variable(name="R", subdomain="pm")],
    equations=[pde("n", "nuc", "2.0"), pde("c", "cyt", "0.0"), pde("e", "ec", "1.0"),
               pde("R", "pm", "0.0", template="surface_pde_with_dilution", source="2.0 * trace(e) * (1.0 - R)")],
    boundary_conditions=[
        BCInterfaceFlux(variable="n", boundary="ne", expression="0.5 * (c - n)"),
        BCInterfaceFlux(variable="c", boundary="ne", expression="0.5 * (n - c)"),
        BCInterfaceFlux(variable="e", boundary="pm", expression="-2.0 * trace(e) * (1.0 - R)"),
    ],
)
many = realize_multi_compartment(three, h=0.1, comm=comm)
solved = integrate_multi_compartment(multi, many, t_final=0.5, rtol=1e-8, atol=1e-10)
for name in ("n", "c", "e", "R"):
    out[f"multi_{name}"] = species_mass(solved, name)
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
    assert parallel["region_inner"] + parallel["region_outer"] == pytest.approx(
        serial["region_inner"] + serial["region_outer"], rel=1e-9
    )
    for run in (serial, parallel):  # nucleus + cytosol exchange; free + bound ligand: each conserved
        assert run["multi_n"] + run["multi_c"] == pytest.approx(2.0 * 3.141592653589793 * 0.09, rel=2e-2)
        assert run["multi_R"] > 0.1
    assert parallel["multi_e"] + parallel["multi_R"] == pytest.approx(serial["multi_e"] + serial["multi_R"], rel=1e-9)
