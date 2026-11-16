"""A cell force balance with species on its members, run in time on a fixed skeleton (ADR 018 step 3b; ADR 020).

The species on the members (the cells' compartments) ride the solved velocity `v` (ADR 020 decision 1): no
slot names the carrier. Each is transported in the conservative lab-frame form `−∫ m (v + w − v_mesh)·∇q`
(decision 2), with `w` its `relative_advection` drift and `v_mesh = 0` here. So it is diluted by `m∇·v`, its
total `∫m` is conserved by the transport, and its natural condition on the cell's boundary is zero total flux
relative to the boundary: `n·(−D∇m + w m) = 0` on a held edge, `n·(−D∇m + (v + w)m) = 0` at a stationary
front. The step is staggered, one pass (decision 3): solve `v` from `mⁿ`, then advance `m` by backward Euler
with that `v`.

The scope is decision 4's:
- **A fixed skeleton.** Each membrane bounding the member is held (`dirichlet v = 0`) or is a stationary front
  (a prescribed velocity of zero). The box faces it touches are walls. So no boundary moves, though the
  material inside does, and the mesh stays fixed.
- **One member.**
- **Species on the member only, with no BC.** A species elsewhere, a membrane species and a region variable
  are refused by name. So is any BC on a member species, stricter than ADR 020 decision 2 allows (a flux BC
  on a held edge would be admissible); its natural condition is zero total flux relative to the boundary.
- **Linear transport in `mⁿ⁺¹`.** A diffusion or drift that reads a species is nonlinear, so backward Euler
  refuses it (`NonlinearTermError`). The crowding cut-offs are stage 5.

The force-balance slots `viscosity`, `bulk_viscosity`, `drag` and `active_stress` may read the species, at
their values at `tⁿ`. Moving the cells is stage 4. Writing `v` to the results bundle and routing through the
runner (decision 5) follow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.cell_force_balance import (
    CellForceBalance,
    ForceBalanceError,
    _is_zero,
    assemble_cell_force_balance,
    build_members_mesh,
    compile_parameters,
    t9_equation,
)
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    REGION_SPACE,
    BCDirichlet,
    MathDescription,
    MotionPrescribedVelocity,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import validate_or_raise


@dataclass
class CellForceBalanceRun:
    """A force balance with species on its member, advanced in time on a fixed skeleton."""

    balance: CellForceBalance
    species: DiscreteProblem
    names: tuple[str, ...]
    t: float = 0.0
    _totals: dict[str, fem.Form] = field(default_factory=dict)

    @property
    def velocity(self) -> fem.Function:
        return self.balance.velocity

    def concentration(self, name: str) -> UflExpr:
        """Species `name` as a UFL expression on the members' mesh (a component of the species Function)."""

        unknown = self.species.unknown
        return unknown if len(self.names) == 1 else unknown[self.names.index(name)]

    def total(self, name: str) -> float:
        """`∫m` over the member (ADR 020 decision 2: conserved by the transport)."""

        if name not in self._totals:
            mesh = self.balance.members.mesh
            self._totals[name] = fem.form(self.concentration(name) * ufl.dx(domain=mesh))
        local = float(np.real(fem.assemble_scalar(self._totals[name])))
        return float(self.balance.members.mesh.comm.allreduce(local, op=MPI.SUM))

    def minimum(self, name: str) -> float:
        """The species' smallest nodal value (ADR 020 decision 6: Galerkin transport can undershoot)."""

        V = self.species.V
        n_owned = V.dofmap.index_map.size_local
        values = self.species.unknown.x.array.reshape(-1, len(self.names))[:n_owned, self.names.index(name)]
        local = float(values.min()) if values.size else np.inf
        return float(V.mesh.comm.allreduce(local, op=MPI.MIN))

    def solve_velocity(self) -> fem.Function:
        """The velocity of the current species (at `self.t`), without advancing them."""

        return self.balance.solve(self.t)

    def step(self, dt: float) -> None:
        """One staggered step: `v` from `mⁿ` at `tⁿ`, then `mⁿ⁺¹` by backward Euler with that `v`."""

        self.balance.solve(self.t)
        self.species.dt.value = dt
        self.species.set_time(self.t + dt)
        self.species.step()
        self.t += dt

    def advance(self, t_final: float, dt: float, on_step: Callable[[CellForceBalanceRun], None] | None = None) -> None:
        """Step until `t_final` (the last step shortened to land on it), calling `on_step` after each. Afterwards
        `velocity` is still the one solved from the previous step's species: call `solve_velocity()` for the
        velocity that pairs with the current species (ADR 020 decision 5: at `tⁿ`, `v` solved from `mⁿ`)."""

        while self.t < t_final - 1e-12 * max(1.0, abs(t_final)):
            self.step(min(dt, t_final - self.t))
            if on_step is not None:
                on_step(self)


def run_cell_force_balance(
    md: MathDescription, description: GeometryDescription, *, h: float, comm: MPI.Comm = MPI.COMM_WORLD
) -> CellForceBalanceRun:
    """Validate `md`, realize `description` at mesh size `h`, and assemble the force balance and its member
    species at `t = 0` (initial conditions set). Advance it with `step` or `advance`."""

    validate_or_raise(md)
    return assemble_cell_force_balance_run(md, realize_multi_compartment(description, h=h, comm=comm))


def assemble_cell_force_balance_run(md: MathDescription, geometry: MultiCompartmentGeometry) -> CellForceBalanceRun:
    """The force balance and its member species on `geometry`, at `t = 0`. `md` must already be valid."""

    eq = t9_equation(md)
    species = _member_species(md, eq, geometry)
    prebuilt = build_members_mesh(md, geometry)
    mesh = prebuilt[0].mesh
    n = len(species)
    V = fem.functionspace(mesh, ("Lagrange", 1) if n == 1 else ("Lagrange", 1, (n,)))
    unknown = fem.Function(V, name=",".join(s.variable for s in species))
    current = {s.variable: (unknown if n == 1 else unknown[k]) for k, s in enumerate(species)}
    time = fem.Constant(mesh, np.float64(0.0))
    balance = assemble_cell_force_balance(md, geometry, prebuilt=prebuilt, fields=current, time=time)

    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    components = [(trial, test)] if n == 1 else [(trial[k], test[k]) for k in range(n)]
    trials = {s.variable: components[k][0] for k, s in enumerate(species)}
    # the species implicit (trials) and the velocity lagged (the solved Function), parameters included
    implicit_symbols = compile_parameters(md, mesh, geometry.sizes, time, {**trials, eq.variable: balance.velocity})
    implicit = CompileContext(mesh=mesh, symbols=implicit_symbols)

    def compiled(text: str) -> UflExpr:
        return compile_expression(parse(text), implicit)

    zero = ufl.as_vector([0.0] * mesh.geometry.dim)
    terms: list[Term] = [Term(TermKind.TIME_DERIVATIVE)]
    for (u, w), s in zip(components, species, strict=True):
        if "advection" in s.terms:  # the validator refuses it (ADR 020 decision 1); a backstop for unvalidated math
            raise ForceBalanceError(f"{s.variable!r} rides {eq.variable!r}; 'advection' on a member species is refused")
        if "diffusion" in s.terms:
            terms.append(Term(TermKind.DIFFUSION, compiled(s.terms["diffusion"]) * ufl.dot(ufl.grad(u), ufl.grad(w))))
        drift = compiled(s.terms["relative_advection"]) if "relative_advection" in s.terms else zero
        # ADR 020 decision 2: the carrier is the material's velocity plus the drift, relative to the mesh (at rest)
        terms.append(Term(TermKind.ADVECTION, -u * ufl.dot(balance.velocity + drift, ufl.grad(w))))
        if "source" in s.terms:
            terms.append(Term(TermKind.SOURCE, compiled(s.terms["source"]) * w))
    dt = fem.Constant(mesh, np.float64(0.0))
    problem = DiscreteProblem(
        variable_name=unknown.name,
        V=V,
        trial=trial,
        test=test,
        dx=ufl.dx(domain=mesh),
        unknown=unknown,
        previous=fem.Function(V),
        dt=dt,
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        time=time,
    )
    _ = problem.bilinear_form  # compose now: a nonlinear transport (the cut-offs) is refused here, not at a step
    initial = CompileContext(
        mesh=mesh,
        symbols=compile_parameters(md, mesh, geometry.sizes, time, {**current, eq.variable: balance.velocity}),
    )
    ics = [compile_expression(parse(s.initial_condition or "0"), initial) for s in species]
    problem.interpolate_initial(ics[0] if n == 1 else ufl.as_vector(ics))
    return CellForceBalanceRun(balance=balance, species=problem, names=tuple(s.variable for s in species))


def _member_species(
    md: MathDescription, eq: TemplateEquation, geometry: MultiCompartmentGeometry
) -> list[TemplateEquation]:
    """The species equations of decision 4's scope, after checking the scope."""

    members = eq.members or (eq.subdomain,)
    if len(members) != 1:
        raise ForceBalanceError(
            f"SpeciesNotYetSupported: species with a force balance on several members {list(members)} wait for "
            f"stage 4 (ADR 020 decision 4)"
        )
    (member,) = members
    others = [e for e in md.equations if e is not eq]
    if not others:
        raise ForceBalanceError(
            "a force-balance run needs a species; for the velocity alone use solve_cell_force_balance"
        )
    region = {v.name for v in md.variables if v.space == REGION_SPACE}
    species: list[TemplateEquation] = []
    for other in others:
        if not isinstance(other, TemplateEquation) or other.template != "bulk_radv_diff":
            raise ForceBalanceError(
                f"SpeciesNotYetSupported: {other.variable!r} is governed by {other.template!r}; a run with a force "
                f"balance takes 'bulk_radv_diff' species on the member (ADR 020 decision 4)"
            )
        if other.subdomain != member or other.variable in region:
            raise ForceBalanceError(
                f"SpeciesNotYetSupported: {other.variable!r} lives on {other.subdomain!r}; only species on the "
                f"member {member!r} run with its force balance until stage 5 (ADR 020 decision 4)"
            )
        species.append(other)
    names = {s.variable for s in species}
    for bc in md.boundary_conditions:
        if bc.variable in names:
            raise ForceBalanceError(
                f"SpeciesNotYetSupported: a {bc.kind} BC on {bc.variable!r} at {bc.boundary!r}; a member species "
                f"takes the natural zero-flux condition relative to the cell's boundary (ADR 020 decision 2)"
            )
    held = {
        bc.boundary
        for bc in md.boundary_conditions
        if bc.variable == eq.variable and isinstance(bc, BCDirichlet) and _is_zero(parse(bc.expression))
    }
    motion = {s.name: s.motion for s in md.subdomains}
    for name, membrane in geometry.membranes.items():
        if member not in (membrane.inside, membrane.outside):
            continue
        front = motion.get(name)
        stationary = isinstance(front, MotionPrescribedVelocity) and _is_zero(parse(front.velocity))
        if isinstance(front, MotionPrescribedVelocity) and not stationary:
            raise ForceBalanceError(
                f"MovingSkeleton: the membrane {name!r} is a growing front, so it moves, held or not; species on a "
                f"moving cell wait for stage 4 (ADR 020 decision 4)"
            )
        if name not in held and not stationary:
            raise ForceBalanceError(
                f"MovingSkeleton: the membrane {name!r} is neither held at {eq.variable} = 0 nor a stationary "
                f"front, so it would move; species on a moving cell wait for stage 4 (ADR 020 decision 4)"
            )
        if stationary and "tension" in eq.surfaces.get(name, {}):
            raise ForceBalanceError(f"the stationary front {name!r} carries a tension; a front carries none")
    for bc in md.boundary_conditions:
        if (
            bc.variable == eq.variable
            and bc.boundary in geometry.compartments[member].faces
            and not (isinstance(bc, BCDirichlet) and _is_zero(parse(bc.expression)))
        ):
            raise ForceBalanceError(
                f"MovingSkeleton: the box face {bc.boundary!r} gives {eq.variable!r} a non-zero value; with species "
                f"on the member, a box face it touches is a wall at rest (ADR 020 decision 4)"
            )
    return species


__all__ = ["CellForceBalanceRun", "assemble_cell_force_balance_run", "run_cell_force_balance"]
