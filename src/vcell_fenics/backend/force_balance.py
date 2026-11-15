"""Term builders for the cell force balance, template T9 `cell_force_balance` (ADRs 017–019).

A quasi-static, overdamped, compressible force balance for one vector velocity `v` on the cells' compartments
(the *members*), with loads on the surfaces that bound them. For every test velocity `φ` the residual is

    Σ_members ∫ ( visc(v, φ) + ζ v·φ − P ∇·φ + σ_a ∇·φ ) dx
  + Σ_surfaces ∫ ( γ ∇_Γ·φ − t·φ + Δt γ ∇_Γv : ∇_Γφ ) dS  = 0,

with `visc` either `symmetric`, `2η dev ε(v):ε(φ) + η_b ∇·v ∇·φ` where `dev` is the **3D** deviator in every
dimension (ADR 018 decision 4: in 2D, UFL's `dev` is the 2D deviator, whose kernel holds every conformal map),
or `laplacian`, `η ∇v:∇φ + η_b ∇·v ∇·φ` (ADR 019 decision 3, the operator of Nickaeen et al. 2017). `P` is the
uniform per-cell pressure, `σ_a` an isotropic active stress (contractile positive, ADR 019 decision 2), `γ` a
surface tension and `t` an applied traction (ADR 019 decision 4). The last surface term is the semi-implicit
tension of Dziuk (1991) and Bänsch (2001): the tension form evaluated at the new position `x + Δt v`, using
`∇_Γ·φ = ∇_Γx : ∇_Γφ` (ADR 018 decision 5).

The mesh here is the **members' mesh**: its cells are the members, tagged by member; its facets are tagged by
surface. A surface between a member and a non-member (a free membrane) is part of this mesh's exterior boundary
and is integrated with `ds`; a surface between two members is interior and integrated with `dS`. There `v`, `φ`
and the normal are restricted to the `'+'` side: for a continuous `v` and `φ` the tangential gradient is the
same from both sides (ADR 018 decision 6), and `n⊗n` does not depend on the normal's orientation. A
coefficient on a contact is the caller's: an unrestricted one is read on the `'+'` side, so it must be
single-valued there (a constant, a region variable, a continuous field); one that differs between the sides
is passed already restricted, e.g. `ufl.avg(g)` or `g('+') + g('-')`. Each term is one tagged `Term` of the
`DiscreteProblem` IR (ADR 004) with its own measure, lowered by the `QuasiStatic` scheme. Building the members'
mesh from a realized geometry and the coefficients from a MathDescription is the assembler's job (ADR 018
staging step 3); this module takes UFL coefficients.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh, MeshTags, exterior_facet_indices
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.discrete import DiscreteProblem, QuasiStatic, Term, TermKind

ViscousOperator = Literal["symmetric", "laplacian"]


@dataclass(frozen=True)
class MemberLaw:
    """The volume coefficients of one member (one cell compartment), keyed by its cell tag: the `volumes:` map
    of a T9 equation. Each is a UFL scalar; None means the slot is absent (zero)."""

    tag: int
    viscosity: UflExpr
    bulk_viscosity: UflExpr | None = None
    drag: UflExpr | None = None
    pressure: UflExpr | None = None
    active_stress: UflExpr | None = None


@dataclass(frozen=True)
class SurfaceLaw:
    """The loads on one surface, keyed by its facet tag: the `surfaces:` map. `tension` is a UFL scalar,
    `traction` a UFL vector (force per area on the cell); None means absent."""

    tag: int
    tension: UflExpr | None = None
    traction: UflExpr | None = None


@dataclass(frozen=True)
class MembersMesh:
    """The members' mesh: cells tagged by member, facets tagged by surface."""

    mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags


def deviatoric_inner(a: UflExpr, b: UflExpr) -> UflExpr:
    """`dev a : b` with `dev` the 3D deviator, for symmetric `a`, `b` in 2D or 3D.

    A 2D tensor is the in-plane block of a planar 3D one whose `zz` entry is 0, so `dev₃ a : b =
    a : b − (tr a)(tr b)/3` in either dimension (ADR 018 decision 4). Not `ufl.dev`, which in 2D subtracts
    `tr/2`: that deviator vanishes on a uniform dilation, so a form built from it is not coercive with
    `η_b = 0`."""

    return ufl.inner(a, b) - ufl.tr(a) * ufl.conj(ufl.tr(b)) / 3


def surface_grad(u: UflExpr, n: UflExpr) -> UflExpr:
    """The tangential gradient `∇_Γu = ∇u (I − n⊗n)` of a vector field `u` on a surface with unit normal `n`."""

    identity = ufl.Identity(len(n))
    return ufl.dot(ufl.grad(u), identity - ufl.outer(n, n))


def surface_div(u: UflExpr, n: UflExpr) -> UflExpr:
    """The tangential divergence `∇_Γ·u = tr(∇_Γu)`."""

    return ufl.tr(surface_grad(u, n))


def surface_is_interior(members: MembersMesh, tag: int) -> bool:
    """Whether surface `tag` lies between two members (interior facets of the members' mesh, `dS`) rather than
    on a free membrane (exterior facets, `ds`). Raises if the tag has facets of both kinds, or none."""

    mesh = members.mesh
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    # owned facets only: `exterior_facet_indices` returns owned facets, so a ghost facet of a free membrane
    # would otherwise count as interior (and every facet is owned by exactly one rank)
    facets = members.facet_tags.find(tag)
    facets = facets[facets < mesh.topology.index_map(fdim).size_local]
    exterior = np.isin(facets, exterior_facet_indices(mesh.topology))
    counts = mesh.comm.allreduce(np.array([int(exterior.sum()), int((~exterior).sum())]), op=MPI.SUM)
    n_exterior, n_interior = int(counts[0]), int(counts[1])
    if n_exterior and n_interior:
        raise ValueError(f"surface {tag} has both exterior and interior facets of the members' mesh")
    if not (n_exterior or n_interior):
        raise ValueError(f"surface {tag} has no facets on the members' mesh")
    return n_interior > 0


def force_balance_terms(
    members: MembersMesh,
    v: UflExpr,
    phi: UflExpr,
    member_laws: Sequence[MemberLaw],
    surface_laws: Sequence[SurfaceLaw],
    *,
    viscous_operator: ViscousOperator = "symmetric",
    semi_implicit_dt: UflExpr | None = None,
) -> tuple[Term, ...]:
    """The tagged terms of the T9 residual for trial `v` and test `phi`.

    One `VISCOUS_STRESS` (or `VISCOUS_LAPLACIAN`) term per member, a `DRAG`, `CELL_PRESSURE` and
    `ACTIVE_STRESS` term where the member has that slot; a `SURFACE_TENSION` and `SURFACE_TRACTION` term where
    the surface has that slot, and a `TENSION_SEMI_IMPLICIT` term for each tension when `semi_implicit_dt` is
    given (the semi-implicit scheme of ADR 018 decision 5)."""

    mesh = members.mesh
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=members.cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=members.facet_tags)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=members.facet_tags)
    n = ufl.FacetNormal(mesh)
    terms: list[Term] = []
    for law in member_laws:
        measure = dx(law.tag)
        if viscous_operator == "symmetric":
            viscous = 2 * law.viscosity * deviatoric_inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(phi)))
            kind = TermKind.VISCOUS_STRESS
        elif viscous_operator == "laplacian":
            viscous = law.viscosity * ufl.inner(ufl.grad(v), ufl.grad(phi))
            kind = TermKind.VISCOUS_LAPLACIAN
        else:
            raise ValueError(f"unknown viscous operator {viscous_operator!r}")
        if law.bulk_viscosity is not None:
            viscous = viscous + law.bulk_viscosity * ufl.inner(ufl.div(v), ufl.div(phi))
        terms.append(Term(kind, viscous, measure))
        if law.drag is not None:
            terms.append(Term(TermKind.DRAG, law.drag * ufl.inner(v, phi), measure))
        if law.pressure is not None:
            terms.append(Term(TermKind.CELL_PRESSURE, -ufl.inner(law.pressure, ufl.div(phi)), measure))
        if law.active_stress is not None:
            terms.append(Term(TermKind.ACTIVE_STRESS, ufl.inner(law.active_stress, ufl.div(phi)), measure))
    for surface in surface_laws:
        interior = surface_is_interior(members, surface.tag)
        measure = dS(surface.tag) if interior else ds(surface.tag)
        side = _restrict if interior else _unrestricted
        vs, phis, ns = side(v), side(phi), side(n)
        if surface.tension is not None:
            gamma = _coefficient_on(surface.tension, interior)
            terms.append(Term(TermKind.SURFACE_TENSION, ufl.inner(gamma, surface_div(phis, ns)), measure))
            if semi_implicit_dt is not None:
                stiffness = semi_implicit_dt * gamma * ufl.inner(surface_grad(vs, ns), surface_grad(phis, ns))
                terms.append(Term(TermKind.TENSION_SEMI_IMPLICIT, stiffness, measure))
        if surface.traction is not None:
            if interior:
                raise ValueError(
                    f"surface {surface.tag} lies between two members; a traction is a load on a free membrane "
                    f"(ADR 019 decision 4)"
                )
            terms.append(Term(TermKind.SURFACE_TRACTION, -ufl.inner(surface.traction, phi), measure))
    return tuple(terms)


def _restrict(expr: UflExpr) -> UflExpr:
    return expr("+")


def _unrestricted(expr: UflExpr) -> UflExpr:
    return expr


def _coefficient_on(coefficient: UflExpr, interior: bool) -> UflExpr:
    """A surface coefficient as the form reads it: on a contact, restricted to `'+'` unless the caller already
    restricted it (a two-sided value such as `ufl.avg(g)`)."""

    if not interior or _is_restricted(coefficient):
        return coefficient
    return coefficient("+")


def _is_restricted(expr: UflExpr) -> bool:
    from ufl.corealg.traversal import unique_pre_traversal
    from ufl.restriction import Restricted

    return isinstance(expr, ufl.core.expr.Expr) and any(
        isinstance(node, Restricted) for node in unique_pre_traversal(expr)
    )


@dataclass(frozen=True)
class VelocityCondition:
    """A strong condition `v = g` on a wall or a membrane: the BC, the Function holding `g`, and the Expression
    it is interpolated from (None for zero), which `DiscreteProblem.set_time` re-interpolates."""

    bc: fem.DirichletBC
    value: fem.Function
    expression: fem.Expression | None


def velocity_dirichlet(
    space: fem.FunctionSpace, facet_tags: MeshTags, tag: int, value: UflExpr | None = None
) -> VelocityCondition:
    """A strong condition `v = g` on the facets tagged `tag`: a wall (`g = 0`) or a membrane's given velocity
    (ADR 019 decision 4). `value` is a UFL vector (it may read a time Constant), None for zero."""

    mesh = space.mesh
    fdim = mesh.topology.dim - 1
    dofs = fem.locate_dofs_topological(space, fdim, facet_tags.find(tag))
    g = fem.Function(space)
    expression = None
    if value is not None:
        expression = fem.Expression(value, space.element.interpolation_points)
        g.interpolate(expression)
    return VelocityCondition(fem.dirichletbc(g, dofs), g, expression)


def assemble_force_balance(
    members: MembersMesh,
    member_laws: Sequence[MemberLaw],
    surface_laws: Sequence[SurfaceLaw] = (),
    *,
    viscous_operator: ViscousOperator = "symmetric",
    semi_implicit_dt: float | None = None,
    conditions: Sequence[VelocityCondition] = (),
    space: fem.FunctionSpace | None = None,
    degree: int = 1,
    variable_name: str = "v",
) -> DiscreteProblem:
    """The T9 force balance as a `DiscreteProblem` with the `QuasiStatic` scheme, on a continuous vector
    Lagrange space of `degree` over the members' mesh (`P1` by default, ADR 018 decision 6), or on `space`.
    `semi_implicit_dt` adds the semi-implicit tension with that `Δt` (held in the problem's `dt`); `conditions`
    are the walls and membrane velocities. `step()` solves for `v` on the current geometry.

    Every cell of the members' mesh must belong to a member in `member_laws`: a cell with no viscous term would
    make the operator singular."""

    mesh = members.mesh
    tdim = mesh.topology.dim
    n_owned = mesh.topology.index_map(tdim).size_local
    covered = np.zeros(n_owned, dtype=bool)
    for law in member_laws:
        cells = members.cell_tags.find(law.tag)
        covered[cells[cells < n_owned]] = True
    uncovered = int(mesh.comm.allreduce(int((~covered).sum()), op=MPI.SUM))
    if uncovered:
        raise ValueError(f"{uncovered} cell(s) of the members' mesh belong to no member in member_laws")
    V = space if space is not None else fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
    v, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = fem.Constant(mesh, PETSc.ScalarType(0.0 if semi_implicit_dt is None else semi_implicit_dt))  # type: ignore[operator]
    terms = force_balance_terms(
        members,
        v,
        phi,
        member_laws,
        surface_laws,
        viscous_operator=viscous_operator,
        semi_implicit_dt=None if semi_implicit_dt is None else dt,
    )
    unknown = fem.Function(V, name=variable_name)
    return DiscreteProblem(
        variable_name=variable_name,
        V=V,
        trial=v,
        test=phi,
        dx=ufl.Measure("dx", domain=mesh, subdomain_data=members.cell_tags),
        unknown=unknown,
        previous=fem.Function(V),
        dt=dt,
        terms=terms,
        scheme=QuasiStatic(),
        bcs=[c.bc for c in conditions],
        dirichlet_refreshers=tuple((c.value, c.expression) for c in conditions if c.expression is not None),
    )
