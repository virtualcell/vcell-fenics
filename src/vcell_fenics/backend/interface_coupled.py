"""Backward-Euler assembly for two volume compartments coupled across a membrane (§1.6.2).

Two bulk diffusion species — one per compartment — coupled by **single-sided interface flux** BCs at
their shared membrane: `D ∇u·n = f(trace(u_inner), trace(u_outer), params)` on each side, with the two
sides carrying *independent* `BCInterfaceFlux`es (VCell's in_flux / out_flux are not an enforced
equal-and-opposite balance — each side gets its own Neumann flux). The common case is a permeability
flux `P·(u_outer − u_inner)` into the inner side and its negation into the outer side, which together
drive the two compartments toward equilibrium and conserve mass; `f` may be any expression in the two
interface traces (and, in a follow-up, an adjacent membrane species).

Cross-mesh mechanics (the `InterfaceCoupledGeometry` substrate):

- A block form needs ONE reference mesh that is also an integration domain, so *every* term is
  integrated on the shared **parent** mesh: the per-compartment mass + diffusion over the parent's
  region cells (`dx(parent)(region_tag)`), and the coupling flux over the parent's interior interface
  facets (`dS(parent)(interface_tag)`). Each compartment's P1 function lives on its own submesh and is
  pulled into the parent integral through its `EntityMap`.
- The interface trace of a bulk variable is `membrane_trace(f) = f('+') + f('-')` — each compartment's
  function is non-zero only on its own side, so the sum is its boundary value regardless of the `dS`
  side labelling. Both trials and tests are restricted this way in the coupling term.
- A single-sided permeability flux is **linear in the unknowns**, so it is fully implicit — assembled
  into the block bilinear form via `ufl.extract_blocks`/`ufl.lhs`, with no lagging. The two-block
  system over `MixedFunctionSpace(V_inner, V_outer)` is solved each step.

This module also carries the three-region extension where **surface species on the membrane couple to
both bulks** (the receptor–ligand cell): any number of bulk species per compartment and any number of
surface species, coupled by interface fluxes referencing the surface density and surface reactions
referencing the bulk traces. Both a backward-Euler stepper (`assemble_membrane_coupled`, fully-lagged
/IMEX binding) and a method-of-lines integrator (`integrate_membrane_coupled`, fully-implicit binding
via a **matrix-free** Newton — the exact `∂(binding)/∂(membrane species)` Jacobian block is
un-assemblable in DOLFINx 0.10/0.11, so the implicit Jacobian is applied by finite-differencing the
residual, preconditioned by the assembled partial Jacobian) are provided; the MOL one is
unconditionally stable for stiff binding.

Implemented on this substrate: a moving membrane (prescribed or force-balance velocity, with the mandatory
`ρ ∇_Γ·v_Γ` dilution), in-bulk reactions, and a reservoir Dirichlet on the outer wall. The remaining
follow-up is `BCInterfaceValueEquality` (the `u_inner = k·u_outer` interface constraint).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scifem
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, exterior_facet_indices
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from ufl.algorithms.check_arities import ArityMismatch, check_form_arity

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.diagnostics import NonlinearTermError
from vcell_fenics.backend.geometry import InterfaceCoupledGeometry, membrane_trace
from vcell_fenics.backend.linear_solvers import set_preconditioner
from vcell_fenics.backend.output_times import OutputMonitor
from vcell_fenics.backend.stokes import solve_incompressible_stokes_surface_tension
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    BCNeumann,
    MathDescription,
    ParameterConstant,
    ParameterExpression,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import validate_or_raise


@dataclass
class InterfaceCoupledProblem:
    """A two-bulk interface-coupled solve, advanced by backward-Euler `step()`s. `inner` / `outer` are
    the per-compartment solution `Function`s (on their own submeshes); `_advance` is the per-step
    closure the assembler builds over the block system."""

    inner_var: str
    outer_var: str
    inner: fem.Function
    outer: fem.Function
    _advance: Callable[[], None]

    def step(self) -> None:
        self._advance()

    def field(self, name: str) -> fem.Function:
        if name == self.inner_var:
            return self.inner
        if name == self.outer_var:
            return self.outer
        raise KeyError(f"{name!r} is not a variable of this coupled problem")

    def total_mass(self) -> float:
        """∫u over both compartments (conserved by an equal-and-opposite interface-flux pair, no external flux)."""
        return _total_mass(self.inner, self.outer)


@dataclass
class InterfaceCoupledResult:
    """The result of a **method-of-lines** integration of an interface-coupled system to `t_final`
    (PETSc TS adaptive BDF): the integrated per-compartment solution `Function`s, the step count, and
    the final time. Unlike `InterfaceCoupledProblem` (backward-Euler, stepped) the solve is one call."""

    inner: fem.Function
    outer: fem.Function
    steps: int
    time: float
    fields: dict[str, fem.Function] | None = None  # every species by name (several per compartment)

    def total_mass(self) -> float:
        return _total_mass(self.inner, self.outer)


def _accumulate_ghosts(vector: PETSc.Vec) -> PETSc.Vec:
    """Finalise an assembled vector: add the contributions assembled into ghost entries onto their
    owning ranks. ``petsc.assemble_vector`` leaves this to the caller ("the returned vector is not
    finalised"); without it every contribution at a partition-boundary dof is lost under MPI — the
    coupled solvers leaked ~10% of their mass at n=2 before this. A no-op in serial (no ghosts)."""

    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
    return vector


def _total_mass(inner: fem.Function, outer: fem.Function) -> float:
    """∫u over both compartments (conserved by an equal-and-opposite interface-flux pair, no external flux)."""
    return sum(_mass_of(field) for field in (inner, outer))


def _mass_of(field: fem.Function) -> float:
    """∫field over its own mesh (an area-integral for a surface field, a volume-integral for a bulk one)."""
    mesh = field.function_space.mesh
    local = fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local.real, op=MPI.SUM))


@dataclass
class _MembraneCoupledFields:
    """The shared state of a three-region bulk–surface–bulk solve: any number of bulk species in each of
    two compartments and any number of surface species on their shared membrane. `inner` / `outer` /
    `membrane` are the per-region **vector** `Function`s whose components are the species named in
    `inner_species` / `outer_species` / `membrane_species` order, with by-name accessors."""

    inner_species: list[str]
    outer_species: list[str]
    membrane_species: list[str]
    inner: fem.Function
    outer: fem.Function
    membrane: fem.Function

    def _locate(self, name: str) -> tuple[fem.Function, int]:
        """The (vector `Function`, component index) holding species `name`."""
        for fn, names in (
            (self.inner, self.inner_species),
            (self.outer, self.outer_species),
            (self.membrane, self.membrane_species),
        ):
            if name in names:
                return fn, names.index(name)
        raise KeyError(f"{name!r} is not a variable of this membrane-coupled solve")

    def field(self, name: str) -> fem.Function:
        """A scalar `Function` holding the named species' values (a copy, for inspection)."""
        fn, k = self._locate(name)
        space = fn.function_space
        if space.dofmap.index_map_bs == 1:  # a one-component region has no `sub(0)` to collapse
            out = fem.Function(fem.functionspace(space.mesh, ("Lagrange", 1)), name=name)
            out.x.array[:] = fn.x.array
            return out
        sub_space, dofs = space.sub(k).collapse()
        out = fem.Function(sub_space, name=name)
        out.x.array[:] = fn.x.array[dofs]
        return out

    def mass(self, name: str) -> float:
        """∫ of the named species over its own mesh."""
        fn, k = self._locate(name)
        mesh = fn.function_space.mesh
        local = fem.assemble_scalar(fem.form(fn[k] * ufl.dx(domain=mesh)))
        return float(mesh.comm.allreduce(local.real, op=MPI.SUM))

    def total_mass(self) -> float:
        """∫ of every species over all three meshes — the conserved total for a single-substance model
        (free in each compartment + membrane-bound), since binding only moves it between those pools."""
        return sum(self.mass(n) for n in (*self.inner_species, *self.outer_species, *self.membrane_species))


@dataclass
class MembraneCoupledProblem(_MembraneCoupledFields):
    """A membrane-coupled solve advanced by **backward-Euler** `step()`s. `_advance` is the per-step
    closure over the three-block system."""

    _advance: Callable[[], None]

    def step(self) -> None:
        self._advance()


@dataclass
class MembraneCoupledResult(_MembraneCoupledFields):
    """The result of a **method-of-lines** integration of a membrane-coupled system to `t_final` (PETSc
    TS adaptive BDF with a matrix-free Newton inner solve): the integrated per-region solution `Function`s,
    the step count, and the final time. The solve is one call (vs `MembraneCoupledProblem`'s stepping)."""

    steps: int
    time: float


def _param_symbols(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    """The model's parameters as symbols on `mesh`: constants as `fem.Constant`, expressions compiled in
    declaration order (the import keeps them dependency-ordered) — so a VCell unit factor like
    `KFlux = AreaPerUnitArea/VolumePerUnitVolume` or `UnitFactor = pow(KMOLE, 1)` binds as a UFL
    expression, not only bare constants. Mirrors the single-mesh `_compile_context` parameter loop."""
    scratch: dict[str, UflExpr] = {
        "geom.x": ufl.SpatialCoordinate(mesh),
        "sim.t": fem.Constant(mesh, PETSc.ScalarType(0.0)),  # type: ignore[operator]
    }
    ctx = CompileContext(mesh=mesh, symbols=scratch)
    params: dict[str, UflExpr] = {}
    for p in md.parameters:
        if isinstance(p, ParameterConstant):
            value: UflExpr = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
        elif isinstance(p, ParameterExpression):
            value = compile_expression(parse(p.expression), ctx)
        else:
            raise NotImplementedError(f"coupled assembly cannot bind parameter {p.name!r} of type {type(p).__name__}")
        scratch[p.name] = params[p.name] = value  # later parameters may reference this one
    return params


class _SweptMeasures:
    """Per-cell swept-measure tracking for the conservative ALE coupled forms, across the moving meshes.

    For each named mesh it holds two DG0 fields, refreshed by `capture()` (before a move) + `update(dt)`
    (after): `volume_ratio` `= |Kⁿ|/|Kⁿ⁺¹|` and `effective_dilution` `= ln(|Kⁿ⁺¹|/|Kⁿ|)/dt`. The P1 local
    mass matrix scales linearly with cell volume (a bulk) or facet length/area (a membrane), so these are
    the *exact* per-cell swept quantities — the same guarantee `backend/discrete._MeshMotion` gives the
    single-mesh paths, here for the two bulks + the membrane at once. `volume_ratio` drives the
    backward-Euler conservative time term (`(u − r·uⁿ)·w`); `effective_dilution` drives the strided-MOL
    dilution so its continuous over-a-stride decay cancels the discrete mesh jump. Both are 1.0 / 0.0
    before the first move."""

    def __init__(self, meshes: dict[str, Mesh]) -> None:
        self._forms: dict[str, fem.Form] = {}
        self.volume_ratio: dict[str, fem.Function] = {}
        self.effective_dilution: dict[str, fem.Function] = {}
        self._before: dict[str, Any] = {}
        for key, mesh in meshes.items():
            dg0 = fem.functionspace(mesh, ("DG", 0))
            self._forms[key] = fem.form(ufl.TestFunction(dg0) * ufl.dx)
            ratio = fem.Function(dg0)
            ratio.x.array[:] = 1.0
            self.volume_ratio[key] = ratio
            self.effective_dilution[key] = fem.Function(dg0)

    def capture(self) -> None:
        """Record each mesh's per-cell measures on the pre-move configuration."""
        self._before = {key: fem.assemble_vector(form).array.copy() for key, form in self._forms.items()}

    def update(self, dt: float) -> None:
        """After the move, set `volume_ratio = |Kⁿ|/|Kⁿ⁺¹|` and `effective_dilution = ln(|Kⁿ⁺¹|/|Kⁿ|)/dt`."""
        for key, form in self._forms.items():
            after = fem.assemble_vector(form).array
            before = self._before[key]
            self.volume_ratio[key].x.array[:] = before / after
            self.effective_dilution[key].x.array[:] = np.log(after / before) / dt


class MembraneCoupledMeshMotion:
    """Advances the three-region substrate (parent + two bulk submeshes + membrane) under a **prescribed
    membrane velocity**, conservative-ALE style. The membrane moves by `dt·v`; the parent's interior
    follows by a **harmonic extension** (∇²d = 0 with d = `dt·v` on the membrane facets and d = 0 on the
    exterior box/wall — so the cell deforms while the outer boundary stays put); the two bulk submeshes
    inherit the parent's displacement at their (coincident) nodes. All four meshes therefore move by the
    *same* field at coincident nodes, so the **topological** `EntityMap`s stay valid and the coupling
    form re-assembles on the deformed geometry.

    `parent_displacement` is `dt·v_mesh` on the parent — its `∇·` is the coefficient of the bulk dilution
    `L ∇·v_mesh` (the volumetric analogue); `membrane_velocity` is `v` on the membrane mesh, whose
    surface divergence `∇_Γ·v_Γ` is the **mandatory** surface dilution `ρ ∇_Γ·v_Γ`. Both bulk and surface
    fields co-move with the deforming domain, so the membrane flux is the ordinary diffusive one (no
    moving-boundary relative-flux correction) and total ligand is conserved across the motion."""

    def __init__(self, md: MathDescription, geometry: InterfaceCoupledGeometry, velocity: str, dt: float) -> None:
        parent = geometry.parent_mesh
        gdim, tdim = parent.geometry.dim, parent.topology.dim
        parent.topology.create_connectivity(tdim - 1, tdim)
        self._gdim = gdim
        v_parent = compile_expression(
            parse(velocity),
            CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **_param_symbols(md, parent)}),
        )

        # Harmonic extension on the parent: ∇²d = 0, d = dt·v on the membrane, d = 0 on the exterior.
        space = fem.functionspace(parent, ("Lagrange", 1, (gdim,)))
        self._disp = fem.Function(space)
        self._interface_disp = fem.Function(space)
        self._interface_expr = fem.Expression(dt * v_parent, space.element.interpolation_points)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        rhs = ufl.inner(fem.Constant(parent, np.zeros(gdim, dtype=PETSc.ScalarType)), test) * ufl.dx
        interface_dofs = fem.locate_dofs_topological(space, tdim - 1, geometry.facet_tags.find(geometry.interface_tag))
        exterior_dofs = fem.locate_dofs_topological(space, tdim - 1, exterior_facet_indices(parent.topology))
        self._problem = LinearProblem(
            a,
            rhs,
            u=self._disp,
            bcs=[
                fem.dirichletbc(self._interface_disp, interface_dofs),
                fem.dirichletbc(fem.Function(space), exterior_dofs),
            ],
            petsc_options_prefix=f"vcellfenics_membrane_motion_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        # The parent + both bulk submeshes move by the harmonic displacement. The node→dof permutation
        # (cKDTree match of each mesh's nodes to the displacement dofs; coincident submesh nodes match
        # exactly) is TOPOLOGICAL and fixed, so it is computed ONCE here on the initial geometry and reused
        # — querying the moved node positions each step (against the initial dof tree) would mis-match and
        # progressively distort the mesh.
        self._bulk_meshes = (parent, geometry.inner_mesh, geometry.outer_mesh)
        tree = cKDTree(space.tabulate_dof_coordinates())
        self._perms = [tree.query(mesh.geometry.x)[1] for mesh in self._bulk_meshes]

        # The membrane moves by dt·v directly (every node is on the boundary); `v_membrane` feeds the
        # surface dilution.
        membrane = geometry.membrane_mesh
        self.membrane_velocity = compile_expression(
            parse(velocity),
            CompileContext(membrane, {"geom.x": ufl.SpatialCoordinate(membrane), **_param_symbols(md, membrane)}),
        )
        mem_space = fem.functionspace(membrane, ("Lagrange", 1, (gdim,)))
        self._mem_disp = fem.Function(mem_space)
        self._mem_expr = fem.Expression(dt * self.membrane_velocity, mem_space.element.interpolation_points)
        self._mem_perm = cKDTree(mem_space.tabulate_dof_coordinates()).query(membrane.geometry.x)[1]
        self._membrane_mesh = membrane
        # Per-cell swept measures on the field meshes (the two bulks + the membrane), for the conservative
        # ALE forms — refreshed each `advance` by bracketing the move.
        self._dt_value = float(dt)
        self._swept = _SweptMeasures({"in": geometry.inner_mesh, "out": geometry.outer_mesh, "mem": membrane})

    @property
    def parent_displacement(self) -> fem.Function:
        """The parent's nodal displacement this step (= `dt·v_mesh`); `∇·` of it is the bulk dilution
        coefficient `L ∇·v_mesh`."""
        return self._disp

    def volume_ratio(self, region: str) -> fem.Function:
        """Per-cell `|Kⁿ|/|Kⁿ⁺¹|` on the `"in"` / `"out"` bulk or `"mem"` membrane mesh (the conservative
        backward-Euler correction), refreshed each `advance`. 1.0 before the first move."""
        return self._swept.volume_ratio[region]

    def effective_dilution(self, region: str) -> fem.Function:
        """Per-cell GCL-consistent `ln(|Kⁿ⁺¹|/|Kⁿ|)/dt` on the `"in"`/`"out"`/`"mem"` mesh (the strided-MOL
        dilution rate whose continuous decay cancels the discrete mesh jump), refreshed each `advance`."""
        return self._swept.effective_dilution[region]

    def advance(self) -> None:
        self._swept.capture()
        self._interface_disp.interpolate(self._interface_expr)
        self._problem.solve()
        disp = self._disp.x.array.reshape((-1, self._gdim))
        for mesh, perm in zip(self._bulk_meshes, self._perms, strict=True):
            mesh.geometry.x[:, : self._gdim] += disp[perm]
        self._mem_disp.interpolate(self._mem_expr)
        self._membrane_mesh.geometry.x[:, : self._gdim] += self._mem_disp.x.array.reshape((-1, self._gdim))[
            self._mem_perm
        ]
        self._swept.update(self._dt_value)


class ForceBalanceMeshMotion:
    """Advances the three-region substrate under a membrane velocity **solved from a force balance**,
    rather than prescribed: each step solves an incompressible surface-tension Stokes flow on the cyto
    (the cell interior) and moves the substrate by the resulting boundary velocity. This is the
    migrating-cell mechanics — the membrane moves under its own tension + bulk pressure, no scripted
    field — and it is a drop-in for `MembraneCoupledMeshMotion` in the coupled solvers (same
    `parent_displacement` / `membrane_velocity` / `advance()` interface).

    Mechanics. `solve_incompressible_stokes_surface_tension` on the cyto gives a divergence-free velocity
    `v` (Taylor–Hood) whose boundary trace is the membrane velocity; the tension γ enters as the
    Laplace–Beltrami continuous-surface-force load (no explicit curvature). A *circle* of uniform γ is the
    Laplace fixed point (v ≈ 0, pressure γ/R), so it stays put; a deformed cell relaxes toward the
    minimal-perimeter circle. Because the flow is incompressible, `∮ v·n = 0` and the cell area is conserved.

    **Mechano-chemical coupling.** The tension is a field, not a constant: `update_tension_from_receptor`
    sets `γ = base + sensitivity·R` from a membrane species (a receptor density), so the biochemistry
    actively reshapes the cell. A non-uniform `R` breaks the circle's symmetry — the membrane contracts
    harder where γ (and so R) is larger, and the cell migrates (the Marangoni force `∇_Γγ` is carried
    automatically by the Laplace–Beltrami load). Call it in the stepping loop after each `step()`.

    Cross-mesh motion mirrors `MembraneCoupledMeshMotion`: the membrane moves by `dt·v`, the parent
    interior by a harmonic extension of that boundary displacement, and the bulk submeshes inherit the
    parent displacement — so the topological `EntityMap`s stay valid. The cyto velocity (P2) is sampled on
    a P1 space and transferred to the parent interface dofs and the membrane via **fixed** node→dof
    permutations (the cell boundary, parent interface, and membrane nodes are coincident and co-move).
    `membrane_velocity` is the solved boundary velocity as a P1 Function (updated each step), feeding the
    mandatory surface dilution `ρ ∇_Γ·v_Γ`; `parent_displacement` (= dt·v_mesh) feeds the bulk dilution.
    """

    def __init__(
        self,
        geometry: InterfaceCoupledGeometry,
        *,
        tension: float,
        dt: float,
        viscosity: float = 1.0,
        screening: float = 1.0,
        tension_smoothing: float = 0.0,
        area_correction: bool = False,
    ) -> None:
        parent, cyto, membrane = geometry.parent_mesh, geometry.inner_mesh, geometry.membrane_mesh
        gdim, tdim = parent.geometry.dim, parent.topology.dim
        parent.topology.create_connectivity(tdim - 1, tdim)
        self._gdim, self._dt = gdim, dt
        self._cyto, self._membrane = cyto, membrane
        self._viscosity, self._screening = viscosity, screening
        self._tension_smoothing = tension_smoothing
        self._area_correction = area_correction

        # The surface tension as a P1 scalar **field** on the cyto (only its boundary values enter the
        # Stokes load). Uniform by default; `update_tension_from_receptor` drives it from a membrane species
        # for the mechano-chemical case (γ = base + sensitivity·R), making the tension — and so the motion —
        # depend on the biochemistry.
        self._tension_space = fem.functionspace(cyto, ("Lagrange", 1))
        self._tension = fem.Function(self._tension_space)
        self._tension.x.array[:] = tension

        # The cyto velocity (Taylor–Hood P2) is sampled on this P1 vector space each step for node transfer.
        self._v1 = fem.Function(fem.functionspace(cyto, ("Lagrange", 1, (gdim,))))

        # Harmonic extension on the parent: ∇²d = 0, d = dt·v on the membrane facets, d = 0 on the exterior.
        space = fem.functionspace(parent, ("Lagrange", 1, (gdim,)))
        self._disp = fem.Function(space)
        self._interface_disp = fem.Function(space)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        rhs = ufl.inner(fem.Constant(parent, np.zeros(gdim, dtype=PETSc.ScalarType)), test) * ufl.dx
        self._idofs = fem.locate_dofs_topological(space, tdim - 1, geometry.facet_tags.find(geometry.interface_tag))
        exterior_dofs = fem.locate_dofs_topological(space, tdim - 1, exterior_facet_indices(parent.topology))
        self._problem = LinearProblem(
            a,
            rhs,
            u=self._disp,
            bcs=[
                fem.dirichletbc(self._interface_disp, self._idofs),
                fem.dirichletbc(fem.Function(space), exterior_dofs),
            ],
            petsc_options_prefix=f"vcellfenics_force_balance_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        # The boundary velocity as a P1 Function on the membrane mesh — feeds the surface dilution.
        self.membrane_velocity = fem.Function(fem.functionspace(membrane, ("Lagrange", 1, (gdim,))))

        # FIXED node→dof permutations (all topological): the bulk meshes follow the parent harmonic
        # displacement; the parent interface dofs and the membrane dofs sample the cyto velocity. The cell
        # boundary, parent interface, and membrane nodes are coincident and co-move, so these never change.
        self._bulk_meshes = (parent, cyto, geometry.outer_mesh)
        bulk_tree = cKDTree(space.tabulate_dof_coordinates())
        self._perms = [bulk_tree.query(mesh.geometry.x)[1] for mesh in self._bulk_meshes]
        cyto_tree = cKDTree(self._v1.function_space.tabulate_dof_coordinates())
        self._iface_to_cyto = cyto_tree.query(space.tabulate_dof_coordinates()[self._idofs])[1]
        mem_dof_x = self.membrane_velocity.function_space.tabulate_dof_coordinates()
        self._mem_to_cyto = cyto_tree.query(mem_dof_x)[1]
        self._mem_perm = cKDTree(mem_dof_x).query(membrane.geometry.x)[1]
        # Per-cell swept measures on the field meshes (cyto = inner, outer, membrane) for the conservative
        # ALE forms — refreshed each `advance` by bracketing the move.
        self._dt_value = float(dt)
        self._swept = _SweptMeasures({"in": cyto, "out": geometry.outer_mesh, "mem": membrane})
        # Membrane (scalar P1) dofs → cyto tension dofs, for transferring a surface species onto the tension.
        mem_scalar = fem.functionspace(membrane, ("Lagrange", 1))
        self._mem_to_tension = cKDTree(self._tension_space.tabulate_dof_coordinates()).query(
            mem_scalar.tabulate_dof_coordinates()
        )[1]

        # Tension regularization (default off): an implicit surface-Helmholtz filter on the membrane that
        # smooths the boundary tension over a length `tension_smoothing` BEFORE it drives the Stokes load.
        # The mechano-chemical tension γ = base + α·(a − h) can develop a node-scale spike where the signaling
        # field localizes; the Laplace–Beltrami load then turns that into a node-scale force that lurches a
        # single membrane node out, kinks the front, and the curvature feedback blows up (the leading-edge
        # instability). Cortical tension has a finite spatial correlation length, so a sub-mesh-scale spike in
        # a signaling field should not produce a sub-mesh-scale mechanical force; filtering γ removes the
        # driver while leaving the incompressible `∮ v·n = 0` area conservation exactly intact (we never touch
        # the velocity). The filter solves `(I − ℓ² Δ_Γ) γ̃ = γ` on the membrane curve — mass + ℓ² stiffness,
        # unconditionally stable — damping wavelengths ≲ ℓ and passing the smooth large-scale tension through.
        if tension_smoothing > 0.0:
            self._gamma_raw = fem.Function(mem_scalar)
            self._gamma_smooth = fem.Function(mem_scalar)
            tr, te = ufl.TrialFunction(mem_scalar), ufl.TestFunction(mem_scalar)
            eps = tension_smoothing**2
            a_filter = (tr * te + eps * ufl.inner(ufl.grad(tr), ufl.grad(te))) * ufl.dx
            self._filter_problem = LinearProblem(
                a_filter,
                self._gamma_raw * te * ufl.dx,
                u=self._gamma_smooth,
                petsc_options_prefix=f"vcellfenics_tension_filter_{id(self):x}_",
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )

        # Area conservation (default off) — a divergence-free re-projection of the velocity, applied at the
        # interpolation. The P2 (Taylor–Hood) Stokes velocity is divergence-free, so ∮ u·n = 0 over the cyto
        # boundary to machine precision (the constant pressure test gives ∫∇·u = 0). But the mesh is moved by
        # the P1 interpolation of u, and ∮ (P1 u)·n ≠ 0: interpolating to P1 drops the P2 edge-midpoint velocity,
        # which a straight-edged polygon cannot carry — a spurious inward flux ∝ h that shrinks the cell
        # (~0.13%/step at h=0.06), intrinsic to a P1 ALE polygon. `advance` removes exactly that net flux from
        # the interpolated velocity (a radial α(x − centroid) with α = −∮ v1·n / 2A), restoring ∮ ·n = 0; the
        # area form supplies A and the flux form supplies ∮ v1·n.
        if area_correction:
            self._area_form = fem.form(fem.Constant(cyto, PETSc.ScalarType(1.0)) * ufl.dx)  # type: ignore[operator]
            self._flux_form = fem.form(ufl.dot(self._v1, ufl.FacetNormal(cyto)) * ufl.ds(domain=cyto))

    @property
    def parent_displacement(self) -> fem.Function:
        """The parent's nodal displacement this step (= `dt·v_mesh`); `∇·` of it is the bulk dilution."""
        return self._disp

    def volume_ratio(self, region: str) -> fem.Function:
        """Per-cell `|Kⁿ|/|Kⁿ⁺¹|` on the `"in"`/`"out"`/`"mem"` mesh (conservative BE correction),
        refreshed each `advance`. 1.0 before the first move."""
        return self._swept.volume_ratio[region]

    def effective_dilution(self, region: str) -> fem.Function:
        """Per-cell GCL-consistent `ln(|Kⁿ⁺¹|/|Kⁿ|)/dt` on the `"in"`/`"out"`/`"mem"` mesh (strided-MOL
        dilution rate), refreshed each `advance`."""
        return self._swept.effective_dilution[region]

    @property
    def tension(self) -> fem.Function:
        """The current surface-tension field (a cyto P1 scalar; only boundary values drive the flow)."""
        return self._tension

    def update_tension_from_receptor(self, receptor: fem.Function, *, base: float, sensitivity: float) -> None:
        """Set the membrane tension to ``base + sensitivity·receptor`` — the mechano-chemical law making the
        force balance depend on a surface species (e.g. a receptor density `R`). `receptor` is a membrane
        P1 scalar `Function` (e.g. ``problem.field("R")``); its values are transferred to the coincident
        cyto-boundary tension dofs. Call it in the stepping loop after each `step()` so the next motion uses
        the freshly-solved density. A non-uniform `R` makes the tension non-uniform → the cell migrates."""
        self._tension.x.array[:] = base
        self._tension.x.array[self._mem_to_tension] = base + sensitivity * receptor.x.array

    def advance(self) -> None:
        self._swept.capture()
        # Regularize the tension first (if enabled): smooth its boundary trace along the membrane so a
        # node-scale signaling spike cannot drive the leading-edge curvature instability. Read the current
        # boundary γ onto the membrane, filter, write it back to the coincident cyto-boundary dofs — the
        # interior cyto γ dofs are unused by the (ds-only) Stokes load, so they need not be touched.
        if self._tension_smoothing > 0.0:
            self._gamma_raw.x.array[:] = self._tension.x.array[self._mem_to_tension]
            self._filter_problem.solve()
            self._tension.x.array[self._mem_to_tension] = self._gamma_smooth.x.array
        # Solve the surface-tension force balance on the (current) cyto; sample the velocity on P1.
        velocity, _ = solve_incompressible_stokes_surface_tension(
            self._cyto, tension=self._tension, viscosity=self._viscosity, screening=self._screening
        )  # `self._tension` is a field: uniform, or coupled to a surface species (mechano-chemical)
        self._v1.interpolate(velocity)

        # Re-project the interpolated velocity to be discretely divergence-free (if enabled) — a correction ON
        # the velocity, at the interpolation, not a separate geometric area rescale. The P2 Stokes velocity is
        # divergence-free, so ∮ u·n = 0 over the cyto boundary; but its P1 interpolation (which moves the mesh)
        # is not — ∮ (P1 u)·n drops the P2 edge-midpoint flux that a straight-edged polygon cannot carry, a
        # spurious inward flux ∝ h. Remove exactly that net flux as a radial field α(x − centroid), whose flux
        # ∮ α(x − c)·n = α∫∇·(x − c) = 2αA, so α = −(∮ v1·n)/(2A) makes the corrected velocity satisfy ∮ ·n = 0.
        # The explicit mesh move by it then conserves area to O(dt²). (Assemble the flux off the *uncorrected*
        # v1, then add the correction in place; the downstream interface/membrane motion uses the result.)
        if self._area_correction:
            flux = float(fem.assemble_scalar(self._flux_form).real)  # ∮ v1·n: the spurious interpolation flux
            area_now = float(fem.assemble_scalar(self._area_form).real)
            alpha = -flux / (2.0 * area_now)
            centroid = self._membrane.geometry.x[:, : self._gdim].mean(axis=0)
            coords = self._v1.function_space.tabulate_dof_coordinates()[:, : self._gdim]
            self._v1.x.array.reshape((-1, self._gdim))[:] += alpha * (coords - centroid)
        cyto_v = self._v1.x.array.reshape((-1, self._gdim))

        # Harmonic-extend the (corrected) boundary velocity over the parent, then move parent + bulk submeshes.
        self._interface_disp.x.array.reshape((-1, self._gdim))[self._idofs] = self._dt * cyto_v[self._iface_to_cyto]
        self._problem.solve()
        disp = self._disp.x.array.reshape((-1, self._gdim))
        for mesh, perm in zip(self._bulk_meshes, self._perms, strict=True):
            mesh.geometry.x[:, : self._gdim] += disp[perm]
        # The membrane velocity Function (for the surface dilution) + move the membrane by dt·v.
        self.membrane_velocity.x.array.reshape((-1, self._gdim))[:] = cyto_v[self._mem_to_cyto]
        self._membrane.geometry.x[:, : self._gdim] += (
            self._dt * self.membrane_velocity.x.array.reshape((-1, self._gdim))[self._mem_perm]
        )
        self._swept.update(self._dt_value)


def _sum_forms(terms: list[UflExpr]) -> UflExpr:
    """Sum a non-empty list of UFL form terms (sidesteps `sum()`'s int-start typing)."""
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


_RESERVOIR_PENALTY = 1.0e6
"""Weak-Dirichlet penalty for a reservoir BC: the boundary value is held to ~1/penalty."""


def _reservoir_dirichlet(
    md: MathDescription, geometry: InterfaceCoupledGeometry, outer_species: list[str], ctx: CompileContext
) -> list[tuple[int, UflExpr]]:
    """The outer-wall **reservoir** Dirichlet BCs as `(component index, compiled value)` pairs — a held
    concentration on the outer box boundary that sustains a bulk species against depletion (e.g. a ligand
    reservoir feeding the membrane binding). Enforced **weakly** by a penalty term `β ∮_wall (u − g) w ds`
    rather than a strong constraint, so it composes with the blocked cross-mesh form as an ordinary
    boundary integral (no per-block Dirichlet bookkeeping). Only the **outer** compartment touches the box,
    so a Dirichlet on any other boundary or species is rejected loudly."""
    reservoirs: list[tuple[int, UflExpr]] = []
    for bc in md.boundary_conditions:
        if not isinstance(bc, BCDirichlet):
            continue
        if bc.boundary != geometry.outer:
            raise NotImplementedError(
                f"Dirichlet BC on boundary {bc.boundary!r}: the membrane-coupled solver supports a reservoir "
                f"Dirichlet only on the outer box boundary {geometry.outer!r}"
            )
        if bc.variable not in outer_species:
            raise NotImplementedError(
                f"Dirichlet BC on {bc.variable!r}: only the outer-compartment species {outer_species} reach "
                f"the outer box boundary {geometry.outer!r}"
            )
        reservoirs.append((outer_species.index(bc.variable), compile_expression(parse(bc.expression), ctx)))
    return reservoirs


def assemble_interface_coupled(
    md: MathDescription, geometry: InterfaceCoupledGeometry, *, dt: float
) -> InterfaceCoupledProblem:
    """Assemble a backward-Euler step for two bulk compartments coupled by single-sided interface flux
    BCs (see the module docstring). Returns an `InterfaceCoupledProblem` whose `step()` advances `dt`."""

    validate_or_raise(md)
    equations = {eq.subdomain: eq for eq in md.equations if isinstance(eq, TemplateEquation)}
    inner_eq = _require_equation(equations, geometry.inner_subdomain)
    outer_eq = _require_equation(equations, geometry.outer_subdomain)
    inner_var, outer_var = inner_eq.variable, outer_eq.variable

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1))
    u_in_fn, u_in_prev = fem.Function(V_in, name=inner_var), fem.Function(V_in)
    u_out_fn, u_out_prev = fem.Function(V_out, name=outer_var), fem.Function(V_out)
    n_in = V_in.dofmap.index_map.size_local
    n_out = V_out.dofmap.index_map.size_local

    w_in, w_out = ufl.TestFunction(V_in), ufl.TestFunction(V_out)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)(geometry.interface_tag)
    ds_wall = ufl.Measure("ds", domain=parent, subdomain_data=geometry.facet_tags)(geometry.outer_tag)
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map]

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    d_in = compile_expression(parse(inner_eq.terms["diffusion"]), ctx)
    d_out = compile_expression(parse(outer_eq.terms["diffusion"]), ctx)

    # Backward-Euler *residual* F(u)=0, per compartment, written with the STATE functions as coefficients
    # (not trial functions) — this is deliberate and load-bearing. The interface-flux coupling references
    # the bulk traces on the parent interface `dS` via `membrane_trace(u) = u('+') + u('-')`. That trick is
    # correct only when the traced object is a *coefficient*: a Function's restriction to its non-side is a
    # genuine zero, which selects the right side. For a bilinear (trial × test) form DOLFINx 0.10/0.11
    # instead aliases *both* restrictions of a submesh *argument* back to the same submesh cell, so the
    # assembled coupling matrix is over-counted (~3.4×) — a mixed codim-0/codim-1 restriction-assembly
    # limitation, the same one `_MatrixFreeShiftedJacobian` documents for the membrane-coupled solve. So we
    # assemble the flux from the state coefficients (exact) and never form the coupling as a matrix: the
    # implicit BE Jacobian is applied **matrix-free** (finite-differenced residual, below), preconditioned
    # by the assemblable — over-counted, but only-a-preconditioner — analytic Jacobian.
    residual_of = {
        inner_var: (u_in_fn - u_in_prev) * w_in * dx_in
        + dt * d_in * ufl.dot(ufl.grad(u_in_fn), ufl.grad(w_in)) * dx_in,
        outer_var: (u_out_fn - u_out_prev) * w_out * dx_out
        + dt * d_out * ufl.dot(ufl.grad(u_out_fn), ufl.grad(w_out)) * dx_out,
    }
    test_of = {inner_var: w_in, outer_var: w_out}
    dx_of = {inner_var: dx_in, outer_var: dx_out}
    state_of = {inner_var: u_in_fn, outer_var: u_out_fn}

    # Optional per-compartment in-bulk `source` (∂c/∂t = … + source): evaluated at the region's OWN species
    # (its state Function), so an affine *or* nonlinear source and a purely spatial forcing (a
    # manufactured-solution term) are all handled uniformly by the residual — no lhs/rhs split needed.
    for eq in (inner_eq, outer_eq):
        if "source" in eq.terms:
            src_ctx = CompileContext(parent, {**ctx.symbols, eq.variable: state_of[eq.variable]})
            residual_of[eq.variable] += (
                -dt * compile_expression(parse(eq.terms["source"]), src_ctx) * test_of[eq.variable] * dx_of[eq.variable]
            )

    # Both interface traces are in scope for every flux (a flux on one side may reference either bulk's
    # trace); each `BCInterfaceFlux` then deposits its flux into its OWN side's test function only. Traces
    # are on the *state* functions, so the flux residual assembles exactly (see the note above).
    coupling_symbols = {
        inner_var: membrane_trace(u_in_fn),
        outer_var: membrane_trace(u_out_fn),
        "geom.x": ufl.SpatialCoordinate(parent),
        **params,
    }
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this assembler handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in test_of:
            continue
        coupling_ctx = CompileContext(parent, coupling_symbols)
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        residual_of[bc.variable] += -dt * flux * membrane_trace(test_of[bc.variable]) * ds_int  # var gains the influx

    # Optional outer-wall Dirichlet on the outer compartment (only it touches the box), enforced weakly by a
    # penalty `β ∮_wall (u_out − g) w ds` — pins the solution where a pure interface-flux problem would leave
    # it up to a null-space constant. Same weak treatment as the membrane-coupled reservoir.
    for _, g_value in _reservoir_dirichlet(md, geometry, [outer_var], ctx):
        residual_of[outer_var] += _RESERVOIR_PENALTY * (u_out_fn - g_value) * w_out * ds_wall

    residual_list = [residual_of[inner_var], residual_of[outer_var]]
    residual_form = fem.form(residual_list, entity_maps=emaps)

    # Assemblable analytic Jacobian, ∂F/∂u — its interface-flux blocks are over-counted (see the note), so
    # it serves only as the GMRES preconditioner; the true action comes from the matrix-free operator.
    u_in_trial, u_out_trial = ufl.TrialFunction(V_in), ufl.TrialFunction(V_out)
    trials, states = (u_in_trial, u_out_trial), (u_in_fn, u_out_fn)
    jacobian_form = fem.form(
        [[ufl.derivative(residual_list[i], states[j], trials[j]) for j in range(2)] for i in range(2)],
        entity_maps=emaps,
    )

    _interpolate_ic(u_in_fn, inner_eq, ctx)
    _interpolate_ic(u_out_fn, outer_eq, ctx)
    u_in_prev.x.array[:] = u_in_fn.x.array
    u_out_prev.x.array[:] = u_out_fn.x.array

    state_vec = petsc.create_vector([V_in, V_out], kind="mpi")

    def unpack(x: PETSc.Vec) -> None:
        u_in_fn.x.array[:n_in] = x.array_r[:n_in]
        u_out_fn.x.array[:n_out] = x.array_r[n_in : n_in + n_out]
        u_in_fn.x.scatter_forward()
        u_out_fn.x.scatter_forward()

    def residual(state: PETSc.Vec, _rate: PETSc.Vec, out: PETSc.Vec) -> None:
        unpack(state)
        b = _accumulate_ghosts(petsc.assemble_vector(residual_form, kind="mpi"))
        b.copy(out)
        b.destroy()

    mf = _MatrixFreeShiftedJacobian(residual, state_vec)  # σ=0 ⇒ its action is the plain ∂F/∂u
    sizes = state_vec.getSizes()
    operator = PETSc.Mat().createPython((sizes, sizes), comm=parent.comm)  # type: ignore[arg-type]
    operator.setPythonContext(mf)
    operator.setUp()
    precond = petsc.assemble_matrix(jacobian_form, kind="mpi")  # static geometry ⇒ assemble once
    precond.assemble()
    ksp = PETSc.KSP().create(parent.comm)
    ksp.setOperators(operator, precond)
    ksp.setType("gmres")
    ksp.getPC().setType("lu")  # direct factor of the (approximate) Jacobian ⇒ GMRES converges in a few its
    ksp.setTolerances(rtol=1.0e-10, atol=1.0e-12)
    rate_vec, resid_vec, delta = state_vec.duplicate(), state_vec.duplicate(), state_vec.duplicate()
    rate_vec.set(0.0)

    def advance() -> None:
        # One BE step = solve F(u)=0 (linear) by matrix-free Newton: the exact ∂F/∂u action (matrix-free)
        # with the assembled approximate Jacobian as preconditioner. Linear ⇒ converges in ~1 iteration; the
        # loop is a guard. The convergence threshold is *relative* to the first residual (the weak-Dirichlet
        # penalty, β≈1e6, otherwise pins the absolute residual floor near the loop's own tolerance).
        state_vec.array[:n_in] = u_in_fn.x.array[:n_in]
        state_vec.array[n_in : n_in + n_out] = u_out_fn.x.array[:n_out]
        residual(state_vec, rate_vec, resid_vec)
        # petsc4py types Vec.norm() loosely as float | tuple; with no norm_type it returns a float.
        r0 = float(resid_vec.norm())  # type: ignore[arg-type]
        for _ in range(10):
            if float(resid_vec.norm()) <= 1.0e-9 * r0 + 1.0e-12:  # type: ignore[arg-type]
                break
            mf.set_base(state_vec, rate_vec, 0.0)
            resid_vec.scale(-1.0)
            ksp.solve(resid_vec, delta)
            state_vec.axpy(1.0, delta)
            residual(state_vec, rate_vec, resid_vec)
        unpack(state_vec)
        u_in_prev.x.array[:] = u_in_fn.x.array
        u_out_prev.x.array[:] = u_out_fn.x.array

    return InterfaceCoupledProblem(inner_var, outer_var, u_in_fn, u_out_fn, advance)


def integrate_interface_coupled(
    md: MathDescription,
    geometry: InterfaceCoupledGeometry,
    *,
    t_final: float,
    dt_initial: float | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    output_times: Sequence[float] = (),
    on_output: Callable[[float, fem.Function, fem.Function], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_output_fields: Callable[[float, dict[str, fem.Function]], None] | None = None,
) -> InterfaceCoupledResult:
    """Integrate an interface-coupled two-bulk system to `t_final` with the **method-of-lines**
    integrator (PETSc TS adaptive BDF) — the same strategy as the FV solver and the single-mesh
    `integrate_discrete_problem`, here over the blocked two-mesh system. The coupling flux may be
    nonlinear in the unknowns (the inner Newton handles it); the time error is ≈0 (adaptive).

    `on_output(t, inner, outer)` receives both compartments' solutions at each of `output_times` in
    [0, `t_final`] (snapshot `Function`s valid for the call; t = 0 is the initial condition, which is
    built in here), recorded by interpolation from a `TS`
    monitor so the step sequence is unchanged (`backend/output_times.py`); `on_progress(t)` follows
    every accepted step.

    **Several species per compartment.** Each species is its own scalar P1 block on its compartment's
    mesh — any number in each — and an in-bulk `source` may reference its compartment's other species
    (mass-action reactions within a compartment). ``on_output_fields(t, {species: field})`` receives every
    species at each output time; ``on_output(t, inner, outer)`` is the one-species-each form (it requires
    exactly one species per compartment). The result's `inner`/`outer` are the first species of each;
    `fields` holds them all. With one species each the blocks, their order and so the numbers are those of
    the original two-block solver.

    The residual `F(state, rate) = M·rate + K·state − coupling` is a 2-block form over the two
    compartment spaces (all integrals on the parent: region `dx`, interface `dS`); the exact Jacobian
    `σ ∂F/∂rate + ∂F/∂state` comes from `ufl.derivative`. Both are assembled monolithically
    (`kind="mpi"`) so the TS state is one blocked vector. **Key:** the TS Jacobian holder is built by
    `assemble_matrix` (the full coupling sparsity, including the off-diagonal blocks), not
    `create_matrix` — a mismatched preallocation breaks the in-callback copy.
    """

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    # Region variables (§1.4.2 T5: a well-mixed species, a membrane potential) are one Real unknown each,
    # appended after the field blocks; the field equations are everything else.
    region_eqs = [eq for eq in template_eqs if eq.template == "region_ode"]
    field_eqs = [eq for eq in template_eqs if eq.template != "region_ode"]
    inner_eqs = [eq for eq in field_eqs if eq.subdomain == geometry.inner_subdomain]
    outer_eqs = [eq for eq in field_eqs if eq.subdomain == geometry.outer_subdomain]
    region_homes = {geometry.inner_subdomain, geometry.outer_subdomain, geometry.membrane_subdomain}
    for eq in region_eqs:
        if eq.subdomain not in region_homes:
            raise NotImplementedError(
                f"region variable {eq.variable!r} lives on {eq.subdomain!r}; the two-compartment solver hosts "
                f"them on {sorted(region_homes)}"
            )
    for subdomain, eqs in ((geometry.inner_subdomain, inner_eqs), (geometry.outer_subdomain, outer_eqs)):
        # each compartment carries field species (each a diffusion equation), region variables, or both
        well_mixed_only = not eqs and any(eq.subdomain == subdomain for eq in region_eqs)
        if (not eqs and not well_mixed_only) or any("diffusion" not in eq.terms for eq in eqs):
            raise NotImplementedError(
                f"interface coupling needs a bulk diffusion equation for each species on compartment {subdomain!r}"
            )
    if on_output is not None and (len(inner_eqs) != 1 or len(outer_eqs) != 1):
        raise ValueError("on_output(t, inner, outer) is for one species per compartment; use on_output_fields")

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1))
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)(geometry.interface_tag)
    ds_wall = ufl.Measure("ds", domain=parent, subdomain_data=geometry.facet_tags)(geometry.outer_tag)
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map]

    # One scalar block per species, inner compartment's first (in declaration order), then the outer's.
    eqs = [*inner_eqs, *outer_eqs]
    spaces = [V_in] * len(inner_eqs) + [V_out] * len(outer_eqs)
    dxs = [dx_in] * len(inner_eqs) + [dx_out] * len(outer_eqs)
    states = [fem.Function(V, name=eq.variable) for V, eq in zip(spaces, eqs, strict=True)]
    rates = [fem.Function(V) for V in spaces]
    tests = [ufl.TestFunction(V) for V in spaces]
    sizes = [V.dofmap.index_map.size_local for V in spaces]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    index_of = {eq.variable: k for k, eq in enumerate(eqs)}
    eq_of = {eq.variable: eq for eq in eqs}
    compartment_of = {eq.variable: eq.subdomain for eq in eqs}

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})

    # Side-aware traces on the membrane. `membrane_trace(f) = f('+') + f('-')` selects f's own side only for a
    # *coefficient*; for a submesh *argument* (a test or trial function) DOLFINx 0.10 aliases both
    # restrictions to the same submesh cell, so an interface-flux Jacobian block counts it twice per side
    # (the over-count behind the BE fix, memory `project_interface_flux_overcount`). Weighting each side by an
    # exact 0/1 indicator of the function's own compartment counts it once — identical for coefficients (the
    # residual, and so the solution, is unchanged) and exact for arguments (the Jacobian becomes the true
    # one: Newton converges quadratically, and a Real block's row — dominated by the flux — is right).
    side = {
        geometry.inner_subdomain: _compartment_indicator(parent, geometry.cell_tags, geometry.inner_region_tag),
        geometry.outer_subdomain: _compartment_indicator(parent, geometry.cell_tags, geometry.outer_region_tag),
    }

    def trace_on(subdomain: str, f: UflExpr) -> UflExpr:
        chi = side[subdomain]
        return f("+") * chi("+") + f("-") * chi("-")

    # Region blocks: a Real (one global DOF) on the parent per region variable, integrated over its region
    # (a compartment's dx, or the membrane's dS). A Real has the same value on both sides of the membrane,
    # so on dS it is referenced as u("+") alone.
    real = scifem.create_real_functionspace(parent)
    region_states = [fem.Function(real, name=eq.variable) for eq in region_eqs]
    region_rates = [fem.Function(real) for _ in region_eqs]
    region_tests = [ufl.TestFunction(real) for _ in region_eqs]
    region_on_membrane = [eq.subdomain == geometry.membrane_subdomain for eq in region_eqs]
    region_dx = [
        ds_int if on_membrane else (dx_in if eq.subdomain == geometry.inner_subdomain else dx_out)
        for eq, on_membrane in zip(region_eqs, region_on_membrane, strict=True)
    ]
    region_bulk = {eq.variable: u for eq, u in zip(region_eqs, region_states, strict=True)}
    region_surface = {eq.variable: u("+") for eq, u in zip(region_eqs, region_states, strict=True)}

    # The MOL residual uses the state Functions directly (so the coupling can be nonlinear), with the
    # time derivative ċ = rate (a Function TS supplies). One block per species, all on the parent.
    diffusion = [compile_expression(parse(eq.terms["diffusion"]), ctx) for eq in eqs]
    residual = [
        rates[k] * tests[k] * dxs[k] + diffusion[k] * ufl.dot(ufl.grad(states[k]), ufl.grad(tests[k])) * dxs[k]
        for k in range(len(eqs))
    ]

    # Optional in-bulk `source` (∂c/∂t = … + source), evaluated at the states of the species in its OWN
    # compartment (affine, nonlinear — e.g. mass action between two species there — or a purely spatial
    # manufactured forcing; the inner Newton differences it): F gains −source·w on that region.
    for k, eq in enumerate(eqs):
        if "source" in eq.terms:
            siblings = {name: states[j] for name, j in index_of.items() if compartment_of[name] == eq.subdomain}
            src_ctx = CompileContext(parent, {**ctx.symbols, **region_bulk, **siblings})
            residual[k] += -compile_expression(parse(eq.terms["source"]), src_ctx) * tests[k] * dxs[k]
    # Every species' interface trace is in scope for every flux (a flux on one side may reference any
    # species on either side); each `BCInterfaceFlux` then deposits its flux into its OWN species' block.
    coupling_symbols = {
        **{eq.variable: trace_on(eq.subdomain, states[k]) for k, eq in enumerate(eqs)},
        **region_surface,
        "geom.x": ufl.SpatialCoordinate(parent),
        **params,
    }
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this integrator handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in index_of:
            continue  # (a region variable's flux BCs join its region balance, below)
        coupling_ctx = CompileContext(parent, coupling_symbols)
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        k = index_of[bc.variable]
        residual[k] += -flux * trace_on(eq_of[bc.variable].subdomain, tests[k]) * ds_int

    # Region balances (T5), in the weak form of a Real test w (w ≡ 1 on its region):
    #   ∫_R u̇ w = ∫_R (uniform_rate + region_rate) w + ∫_{∂R} flux w
    # i.e. |R|·u̇ = |R|·uniform_rate + ∫_R region_rate + ∫ flux — the template's (1/|R|)-averaged ODE.
    region_residual: list[UflExpr] = []
    for r, eq in enumerate(region_eqs):
        w, measure = region_tests[r], region_dx[r]
        if region_on_membrane[r]:
            rate_ctx = CompileContext(parent, coupling_symbols)
            w_here, rate_here = w("+"), region_rates[r]("+")
        else:
            siblings = {name: states[j] for name, j in index_of.items() if compartment_of[name] == eq.subdomain}
            rate_ctx = CompileContext(parent, {**ctx.symbols, **region_bulk, **siblings})
            w_here, rate_here = w, region_rates[r]
        form = rate_here * w_here * measure
        for slot in ("uniform_rate", "region_rate"):
            if slot in eq.terms:
                form += -compile_expression(parse(eq.terms[slot]), rate_ctx) * w_here * measure
        for bc in md.boundary_conditions:
            if (
                isinstance(bc, (BCInterfaceFlux, BCNeumann))
                and bc.variable == eq.variable
                and bc.boundary == geometry.interface
            ):
                flux = compile_expression(parse(bc.expression), CompileContext(parent, coupling_symbols))
                form += -flux * w("+") * ds_int
        region_residual.append(form)

    # Optional outer-wall reservoir Dirichlet on outer-compartment species, weak penalty (same as the BE
    # assembler): F gains β(u − g)·w on the box wall.
    outer_names = [eq.variable for eq in outer_eqs]
    for j, g_value in _reservoir_dirichlet(md, geometry, outer_names, ctx):
        k = index_of[outer_names[j]]
        residual[k] += _RESERVOIR_PENALTY * (states[k] - g_value) * tests[k] * ds_wall

    # The region blocks join the system after the field blocks: one list of spaces/states/rates/residuals.
    n_fields = len(eqs)
    spaces = [*spaces, *([real] * len(region_eqs))]
    states = [*states, *region_states]
    rates = [*rates, *region_rates]
    residual = [*residual, *region_residual]
    sizes = [V.dofmap.index_map.size_local for V in spaces]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    n_blocks = len(states)
    shift = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]  # the TS σ
    jacobian = [
        [
            shift * ufl.derivative(residual[i], rates[j]) + ufl.derivative(residual[i], states[j])
            for j in range(n_blocks)
        ]
        for i in range(n_blocks)
    ]
    residual_form = fem.form(residual, entity_maps=emaps)
    jacobian_form = fem.form(jacobian, entity_maps=emaps)

    for state, eq in zip(states[:n_fields], eqs, strict=True):
        _interpolate_ic(state, eq, ctx)
    for state, eq, measure, on_membrane in zip(region_states, region_eqs, region_dx, region_on_membrane, strict=True):
        state.x.array[:] = _region_average(eq, ctx, measure, on_membrane, parent)

    def unpack(x: PETSc.Vec) -> None:
        for k, state in enumerate(states):
            state.x.array[: sizes[k]] = x.array_r[offsets[k] : offsets[k + 1]]
            state.x.scatter_forward()

    def evaluate_residual(_ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        unpack(x)
        for k, rate in enumerate(rates):
            rate.x.array[: sizes[k]] = x_dot.array_r[offsets[k] : offsets[k + 1]]
            # The owned part alone is not enough: cells on a partition boundary read ghost dofs of ċ in the
            # mass term, and stale ghosts there leaked mass under MPI (as `unpack` refreshes the state's).
            rate.x.scatter_forward()
        b = _accumulate_ghosts(petsc.assemble_vector(residual_form, kind="mpi"))
        b.copy(result)
        b.destroy()

    def evaluate_jacobian(
        _ts: PETSc.TS, _t: float, x: PETSc.Vec, _x_dot: PETSc.Vec, sigma: float, mat: PETSc.Mat, _pre: PETSc.Mat
    ) -> None:
        unpack(x)
        shift.value = sigma
        fresh = petsc.assemble_matrix(jacobian_form, kind="mpi")
        fresh.assemble()
        fresh.copy(mat, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        mat.assemble()
        fresh.destroy()

    ts = PETSc.TS().create(parent.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType("bdf")
    state_vec = petsc.create_vector(spaces, kind="mpi")
    for k, state in enumerate(states):
        state_vec.array[offsets[k] : offsets[k + 1]] = state.x.array[: sizes[k]]
    # Jacobian holder via assemble_matrix → the full coupling sparsity (off-diagonal blocks present).
    jacobian_matrix = petsc.assemble_matrix(jacobian_form, kind="mpi")
    jacobian_matrix.assemble()
    ts.setIFunction(evaluate_residual, state_vec.duplicate())
    ts.setIJacobian(evaluate_jacobian, jacobian_matrix, jacobian_matrix)
    # A small startup step (BDF cold-start, see the single-mesh integrator) then adapt.
    ts.setTimeStep(dt_initial if dt_initial is not None else t_final / 1.0e4)
    ts.setMaxTime(t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(atol, rtol)
    ts.setMaxSNESFailures(-1)
    snes = ts.getSNES()
    snes.setUseEW(False)
    # Inner Newton linear solver. The default is a Krylov solve (GMRES) with an incomplete-LU (ILU)
    # preconditioner — the scalable choice (≈O(N) memory and per-iteration cost), as the single-mesh
    # MOL uses and as VCell's CVODE uses SPGMR+ILU. A *direct* sparse LU (`ksp_type="preonly"`,
    # `pc_type="lu"`) is robust for small problems but its 2D fill-in does not scale to large meshes.
    snes.getKSP().setType(ksp_type)
    set_preconditioner(snes.getKSP(), pc_type)  # ILU → block-Jacobi/ILU(0) under MPI
    ts.setFromOptions()

    monitor: OutputMonitor | None = None
    work: PETSc.Vec | None = None
    wants_output = on_output is not None or on_output_fields is not None
    if wants_output or on_progress is not None:
        names = [eq.variable for eq in (*eqs, *region_eqs)]
        snaps = [fem.Function(V, name=name) for V, name in zip(spaces, names, strict=True)]
        work = state_vec.duplicate()

        def emit(t: float, x: PETSc.Vec) -> None:
            for k, snap in enumerate(snaps):
                snap.x.array[: sizes[k]] = x.array_r[offsets[k] : offsets[k + 1]]
                snap.x.scatter_forward()
            if on_output is not None:
                on_output(t, snaps[0], snaps[1])
            if on_output_fields is not None:
                on_output_fields(t, dict(zip(names, snaps, strict=True)))

        monitor = OutputMonitor(
            output_times if wants_output else (),
            t_start=0.0,
            t_final=t_final,
            work=work,
            emit=emit,
            progress=on_progress,
        )
        ts.setMonitor(monitor)

    ts.solve(state_vec)
    unpack(state_vec)
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    if monitor is not None:
        monitor.finish(ts, final_time, state_vec)
    for obj in (ts, state_vec, jacobian_matrix, work):
        if obj is not None:
            obj.destroy()
    fields = {eq.variable: state for eq, state in zip((*eqs, *region_eqs), states, strict=True)}
    return InterfaceCoupledResult(states[0], states[len(inner_eqs)], steps, final_time, fields)


def _compartment_indicator(parent: Mesh, cell_tags: Any, region_tag: int) -> fem.Function:
    """An exact 0/1 DG0 indicator of one compartment's cells on the parent mesh."""

    indicator = fem.Function(fem.functionspace(parent, ("DG", 0)))
    indicator.x.array[:] = 0.0
    indicator.x.array[cell_tags.indices[cell_tags.values == region_tag]] = 1.0
    indicator.x.scatter_forward()
    return indicator


def _region_average(
    eq: TemplateEquation, ctx: CompileContext, measure: ufl.Measure, on_membrane: bool, parent: Mesh
) -> float:
    """A region variable's initial value: its initial condition averaged over its region (exact for a
    constant; a spatially varying VCell initial expression has no single value, so its mean is taken)."""

    assert eq.initial_condition is not None  # the validator requires one (time-dependent)
    ic = compile_expression(parse(eq.initial_condition), ctx)
    one = fem.Constant(parent, PETSc.ScalarType(1.0))  # type: ignore[operator]
    if on_membrane:
        ic = ic("+")
    total = parent.comm.allreduce(fem.assemble_scalar(fem.form(ic * measure)), op=MPI.SUM)
    size = parent.comm.allreduce(fem.assemble_scalar(fem.form(one * measure)), op=MPI.SUM)
    return float(np.real(total) / np.real(size))


def assemble_membrane_coupled(
    md: MathDescription,
    geometry: InterfaceCoupledGeometry,
    *,
    dt: float,
    velocity: str | None = None,
    motion: MembraneCoupledMeshMotion | ForceBalanceMeshMotion | None = None,
) -> MembraneCoupledProblem:
    """Assemble a backward-Euler step for **membrane species coupled to both bulk compartments**: any
    number of bulk diffusion species in each compartment plus any number of surface species on their
    shared membrane, the full receptor–ligand cell (the §1.6.5/§1.6.6 composable pattern at an *internal*
    interface). Each region is one **vector** P1 space whose components are its species.

    The regions couple at the membrane two ways: (1) each surface species' reaction `source` may
    reference the interface trace of any bulk species in either compartment, and (2) each bulk species'
    `BCInterfaceFlux` may reference any same-compartment trace, any adjacent-compartment trace, and any
    membrane species. Every species is in scope via a single symbol table mapping its name to its
    interface representation. The coupling is **fully lagged (IMEX)**: in the membrane-facet terms the
    bulk traces *and* the surface species use previous-step values, so the binding rate is an explicit
    RHS forcing. The SAME rate is subtracted from the bulk(s) (each flux, tested on its own side) and
    added to the surface (each source, tested on the membrane), so the membrane-facet integrals cancel
    termwise and **mass is conserved to round-off** when the modeller writes the fluxes as the negation
    of the matching production. (Keeping the surface species implicit would need an off-diagonal
    bulk-test × membrane-trial block on `dS` — a codim-0 × codim-1 coupling that leaks the cancellation;
    full lagging sidesteps it, at the cost of conditional stability for stiff binding, which the
    method-of-lines follow-up lifts with a Newton solve.)

    Cross-mesh mechanics (extending the two-bulk substrate, §1.6.2): each species' mass + diffusion is
    integrated on the parent's region cells (`dx(parent)(region)`, per component, with its own diffusion)
    and the surface mass + **intrinsic surface** diffusion on the membrane submesh's own `dx` (its `grad`
    is the tangential ∇_Γ). The coupling is integrated on the parent's interior interface facets (`dS`),
    where each bulk trace is the orientation-robust `membrane_trace(u) = u('+') + u('-')` and a membrane
    species is restricted `ρ('+')` (single-valued on the facet). All three submesh functions are pulled
    into the parent integrals through their `EntityMap`s; with the coupling fully lagged the block matrix
    is the static, block-diagonal mass + diffusion (factorised once) and only the RHS changes each step.

    **Moving membrane (`velocity` or `motion` given).** Pass a prescribed membrane velocity expression
    (e.g. ``"[0.3*geom.x[0], 0.3*geom.x[1]]"``), OR a pre-built motion driver via `motion=` — a
    `MembraneCoupledMeshMotion` (prescribed) or a `ForceBalanceMeshMotion` (the membrane velocity SOLVED
    from a surface-tension force balance, so the cell migrates under its own mechanics; build it with
    `dt` equal to this assembler's `dt`). Either way each `step()` first moves the mesh (membrane by `dt·v`,
    parent interior by harmonic extension, bulk submeshes inherited) and then re-assembles + solves on the
    deformed geometry. Because
    every field co-moves with its mesh, each gains a **lagged dilution** forcing so its integral tracks the
    local volume/area change:
      - bulk species: ``− c^n (∇·v_mesh) w`` on the region cells, with `∇·v_mesh` the (ambient) divergence
        of the parent harmonic displacement — material concentrates/dilutes as the compartment shrinks/grows;
      - surface species: ``− ρ^n (∇_Γ·v_Γ) w`` on the membrane, the **mandatory** `ρ ∇_Γ·v_Γ` stretch term.
        Critically `∇_Γ·v_Γ` is the *surface* divergence ``div(v) − n·(∇v)·n`` (`n = CellNormal`), **not**
        `ufl.div(v)`: on the membrane manifold `ufl.div` returns the ambient divergence (e.g. `2a` for
        `v=a·x` vs the true `a`), which silently over-dilutes and loses ~6% of the surface mass per growth.
    Dilution moves no mass between pools (it is per-region), so total ligand stays conserved — to round-off
    from the termwise-cancelling binding, and to O(dt) from the lagged dilution (refines away). The mesh
    moves so the block matrix is **re-assembled and re-factorised every step** (no longer static).

    **In-bulk reactions.** A bulk species may carry a `source` (`∂c/∂t = … + source`) referencing any
    sibling species in its own compartment by name — assembled implicitly into the block matrix. It must be
    **affine** in the unknowns (e.g. a within-compartment `A ⇌ B`); a nonlinear source (a product of
    species) is rejected here with a `NonlinearTermError` pointing at `integrate_membrane_coupled`, whose
    matrix-free Newton handles it.

    Scope (this increment): backward Euler with affine in-bulk reactions, a *prescribed* membrane velocity
    (no force balance). Nonlinear in-bulk reactions (via the MOL) and a moving membrane *under MOL* are the
    method-of-lines integrator's; force-balance velocity is a follow-up.
    """

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    inner_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.inner_subdomain]
    outer_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.outer_subdomain]
    membrane_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.membrane_subdomain]
    if not inner_eqs or not outer_eqs:
        raise NotImplementedError("membrane coupling needs at least one bulk species in each compartment")
    if not membrane_eqs:
        raise NotImplementedError(
            f"membrane coupling needs a surface equation on the membrane subdomain "
            f"{geometry.membrane_subdomain!r} (the species coupled to both bulks)"
        )
    for eq in (*inner_eqs, *outer_eqs):
        if "diffusion" not in eq.terms:
            raise NotImplementedError(f"membrane coupling needs a bulk diffusion equation for {eq.variable!r}")
    inner_species = [eq.variable for eq in inner_eqs]
    outer_species = [eq.variable for eq in outer_eqs]
    membrane_species = [eq.variable for eq in membrane_eqs]

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1, (len(inner_species),)))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1, (len(outer_species),)))
    V_mem = fem.functionspace(geometry.membrane_mesh, ("Lagrange", 1, (len(membrane_species),)))
    u_in_fn, u_in_prev = fem.Function(V_in), fem.Function(V_in)
    u_out_fn, u_out_prev = fem.Function(V_out), fem.Function(V_out)
    rho_fn, rho_prev = fem.Function(V_mem), fem.Function(V_mem)
    n_in = V_in.dofmap.index_map.size_local * V_in.dofmap.index_map_bs
    n_out = V_out.dofmap.index_map.size_local * V_out.dofmap.index_map_bs
    n_mem = V_mem.dofmap.index_map.size_local * V_mem.dofmap.index_map_bs

    mixed = ufl.MixedFunctionSpace(V_in, V_out, V_mem)
    u_in, u_out, rho = ufl.TrialFunctions(mixed)
    w_in, w_out, w_rho = ufl.TestFunctions(mixed)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    dx_mem = ufl.Measure("dx", domain=geometry.membrane_mesh)
    # A fixed quadrature degree on the interface so the shared binding rate integrates *identically*
    # whether it is tested by the bulk side's `membrane_trace(w_in[k])` (a flux) or the surface side's
    # `w_ρ[k]('+')` (a source) — the termwise cancellation that gives mass conservation relies on both
    # forcing integrals using the same quadrature.
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags, metadata={"quadrature_degree": 4})(
        geometry.interface_tag
    )
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map, geometry.membrane_entity_map]

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    surf_params = _param_symbols(md, geometry.membrane_mesh)
    surf_ctx = CompileContext(
        geometry.membrane_mesh, {"geom.x": ufl.SpatialCoordinate(geometry.membrane_mesh), **surf_params}
    )

    # A single block form cannot mix two meshes (each block needs one integration domain), so the
    # residual splits into a LOCAL part (per-mesh mass + diffusion) and a COUPLING part (all on the
    # parent's interface `dS`). With the coupling fully lagged (below) the bilinear form is the
    # block-diagonal local part — static — so the matrix assembles + factorises once and only the RHS
    # changes each step.
    #
    # Local: each species' mass + diffusion + in-bulk reaction, per component with its own diffusivity.
    # Bulk species on the parent region cells; surface species on the membrane submesh (`grad` there is the
    # tangential ∇_Γ). Static membrane, so no dilution term ρ ∇_Γ·v_Γ.
    #
    # In-bulk reaction (a `source` on a *bulk* species, `∂c/∂t = … + source`): compiled IMPLICITLY against
    # the region's trials so it lands in the block matrix (an affine source is unconditionally stable; a
    # nonlinear one — a product of species — is rejected below for this BE path and handed to the MOL). Its
    # symbol table binds every sibling species in the SAME compartment by name, so a within-compartment
    # reaction `A ⇌ B` resolves both. A surface species' `source` stays the membrane *coupling* reaction
    # (below, on `dS`), not a local term.
    terms: list[UflExpr] = []
    for trial, prev, test, eqs, dx, is_bulk in (
        (u_in, u_in_prev, w_in, inner_eqs, dx_in, True),
        (u_out, u_out_prev, w_out, outer_eqs, dx_out, True),
        (rho, rho_prev, w_rho, membrane_eqs, dx_mem, False),
    ):
        region_ctx = ctx if is_bulk else surf_ctx
        react_ctx = (
            CompileContext(parent, {**ctx.symbols, **{e.variable: trial[j] for j, e in enumerate(eqs)}})
            if is_bulk
            else None
        )
        for k, eq in enumerate(eqs):
            _refuse_lab_frame_advection(eq)
            terms.append((trial[k] - prev[k]) * test[k] * dx)
            diffusion = eq.terms.get("diffusion")
            if diffusion is not None:
                d = compile_expression(parse(diffusion), region_ctx)
                terms.append(dt * d * ufl.dot(ufl.grad(trial[k]), ufl.grad(test[k])) * dx)
            advection = eq.terms.get("relative_advection")
            if advection is not None:
                # Species drift relative to the mesh, `w_rel·∇u` (linear in u → implicit; `grad` on the
                # membrane submesh is tangential, so the same term serves bulk and surface species).
                drift = compile_expression(parse(advection), region_ctx)
                terms.append(dt * ufl.dot(drift, ufl.grad(trial[k])) * test[k] * dx)
            source = eq.terms.get("source")
            if is_bulk and source is not None and react_ctx is not None:
                terms.append(-dt * compile_expression(parse(source), react_ctx) * test[k] * dx)
    f_local = sum(terms[1:], terms[0])

    # Reservoir Dirichlet on the outer box wall (a held concentration sustaining a bulk species), enforced
    # weakly by a penalty `β ∮_wall (u − g) w ds` — it is linear in u_out, so `ufl.lhs` routes `β u w ds`
    # into the matrix and `ufl.rhs` the constant `β g w ds` into the RHS, no per-block Dirichlet handling.
    ds_wall = ufl.Measure("ds", domain=parent, subdomain_data=geometry.facet_tags)(geometry.outer_tag)
    for k, g_value in _reservoir_dirichlet(md, geometry, outer_species, ctx):
        f_local += _RESERVOIR_PENALTY * (u_out[k] - g_value) * w_out[k] * ds_wall

    # Coupling on the parent's interior interface facets, **fully lagged (IMEX)**: the bulk interface
    # traces *and* the surface species use previous-step values, so each binding rate is an explicit RHS
    # forcing. The SAME rate is subtracted from a bulk (its `BCInterfaceFlux`, tested on its own side)
    # and added to the surface (a `source`, tested on the membrane), so the membrane-facet integrals
    # cancel termwise and mass is conserved to round-off. (Keeping the surface species implicit would
    # need an off-diagonal bulk-test × membrane-trial block on `dS` — a codim-0 × codim-1 coupling that
    # leaks the cancellation; full lagging sidesteps it, at the cost of conditional stability for stiff
    # binding. The method-of-lines follow-up lifts that with a Newton solve.)
    #
    # One symbol table puts EVERY species in scope at the membrane: each bulk species as its lagged
    # interface trace, each membrane species as its lagged '+' restriction. Any flux or source then
    # references any same-compartment, adjacent-compartment, or membrane species by name.
    coupling_symbols: dict[str, UflExpr] = {"geom.x": ufl.SpatialCoordinate(parent), **params}
    for k, name in enumerate(inner_species):
        coupling_symbols[name] = membrane_trace(u_in_prev[k])
    for k, name in enumerate(outer_species):
        coupling_symbols[name] = membrane_trace(u_out_prev[k])
    for k, name in enumerate(membrane_species):
        coupling_symbols[name] = rho_prev[k]("+")
    coupling_ctx = CompileContext(parent, coupling_symbols)

    # Test-only structural zeros keep every test block present in the coupling RHS, even for a species
    # that no flux/source happens to touch (else `extract_blocks` yields a None block).
    zero = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    structural = (
        [membrane_trace(w_in[k]) for k in range(len(inner_species))]
        + [membrane_trace(w_out[k]) for k in range(len(outer_species))]
        + [w_rho[k]("+") for k in range(len(membrane_species))]
    )
    f_coupling = zero * sum(structural[1:], structural[0]) * ds_int
    for k, eq in enumerate(membrane_eqs):
        source = eq.terms.get("source")
        if source is not None:
            reaction = compile_expression(parse(source), coupling_ctx)  # ∂ρ_k/∂t = … + reaction
            f_coupling += -dt * reaction * w_rho[k]("+") * ds_int
    bulk_test_of = {name: (w_in, k) for k, name in enumerate(inner_species)}
    bulk_test_of.update({name: (w_out, k) for k, name in enumerate(outer_species)})
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this assembler handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in bulk_test_of:
            continue
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        test, k = bulk_test_of[bc.variable]
        f_coupling += -dt * flux * membrane_trace(test[k]) * ds_int

    # Moving membrane (optional): build the mesh-motion driver and the per-region **lagged dilution**
    # forcing so each co-moving field's integral tracks its volume/area change (see the docstring). Bulk:
    # − c^n (∇·v_mesh) w over the region cells, with `∇·v_mesh` the ambient divergence of the parent
    # harmonic displacement (= dt·v_mesh, so div(disp) already carries the dt). Surface: − dt·ρ^n (∇_Γ·v_Γ) w,
    # where ∇_Γ·v_Γ is the SURFACE divergence `div(v) − n·(∇v)·n` (n = CellNormal) — `ufl.div` on the
    # membrane manifold is the *ambient* divergence and would over-dilute by the curvature term.
    if motion is None and velocity is not None:
        motion = MembraneCoupledMeshMotion(md, geometry, velocity=velocity, dt=dt)
    rhs_dilution_form = None
    if motion is not None:
        # Conservative ALE correction (replaces the old lagged `−uⁿ ∇·v` dilution forcing): the mass term's
        # RHS `uⁿ·w` gets `+(r − 1)·uⁿ·w` added per region, making the equation `(u − r·uⁿ)·w = 0` with
        # r = |Kⁿ|/|Kⁿ⁺¹| the exact per-cell (bulk) / per-facet (membrane) swept ratio. The carried field's
        # mass on the moved mesh then equals its pre-move integral exactly, so dilution lives in the changing
        # measure and each field's substance is conserved to solver precision — no O(dt) geometric-
        # conservation-law drift, and the membrane needs no explicit surface-divergence term (the facet-area
        # ratio *is* ∇_Γ·v_Γ). `volume_ratio` is refreshed by `motion.advance()` before the RHS assembles.
        r_in, r_out, r_mem = motion.volume_ratio("in"), motion.volume_ratio("out"), motion.volume_ratio("mem")
        dilution_terms: list[UflExpr] = []
        for k in range(len(inner_species)):
            dilution_terms.append((r_in - 1.0) * u_in_prev[k] * w_in[k] * dx_in)
        for k in range(len(outer_species)):
            dilution_terms.append((r_out - 1.0) * u_out_prev[k] * w_out[k] * dx_out)
        for k in range(len(membrane_species)):
            dilution_terms.append((r_mem - 1.0) * rho_prev[k] * w_rho[k] * dx_mem)
        rhs_dilution_form = fem.form(ufl.extract_blocks(_sum_forms(dilution_terms)), entity_maps=emaps)

    # Block-diagonal mass + diffusion (+ any affine in-bulk reaction). Static unless the membrane moves, in
    # which case the geometry — and so this matrix — is re-assembled and re-factorised every step (below).
    # A nonlinear in-bulk source makes the bilinear part non-affine: catch the arity mismatch in pure UFL
    # (before form compilation, so we never poison the FFCx cache) and name the fix — the MOL integrator.
    lhs_blocks = ufl.extract_blocks(ufl.lhs(f_local))
    try:
        # Per-BLOCK arity (not the monolithic mixed form: each per-mesh integral carries only a subset of
        # the mixed arguments, which the whole-form checker misreads as a mismatch). A single-space block
        # with a species appearing quadratically (a nonlinear source) trips the mismatch here.
        for row in lhs_blocks:
            for block in row if isinstance(row, list | tuple) else (row,):
                if block is not None:
                    check_form_arity(block, block.arguments())
    except ArityMismatch as nonlinear:
        raise NonlinearTermError(
            "an in-bulk reaction `source` is nonlinear in the species (e.g. a product like A*B) — the "
            "backward-Euler membrane-coupled assembler can only lower a source affine in the unknowns. Use "
            "`integrate_membrane_coupled` (its matrix-free Newton handles nonlinear reactions), or "
            "linearise / lag the term."
        ) from nonlinear
    a_form = fem.form(lhs_blocks, entity_maps=emaps)
    rhs_local_form = fem.form(ufl.extract_blocks(ufl.rhs(f_local)), entity_maps=emaps)
    rhs_coupling_form = fem.form(ufl.extract_blocks(ufl.rhs(f_coupling)), entity_maps=emaps)
    matrix = petsc.assemble_matrix(a_form)
    matrix.assemble()
    ksp = PETSc.KSP().create(parent.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    for fn, eqs, region_ctx in (
        (u_in_fn, inner_eqs, ctx),
        (u_out_fn, outer_eqs, ctx),
        (rho_fn, membrane_eqs, surf_ctx),
    ):
        _interpolate_component_ics(fn, eqs, region_ctx)
    for fn, prev in ((u_in_fn, u_in_prev), (u_out_fn, u_out_prev), (rho_fn, rho_prev)):
        prev.x.array[:] = fn.x.array
    solution = matrix.createVecRight()
    matrix_cell = [matrix]  # mutable holder so a moving-mesh step can swap in the re-assembled matrix

    def advance() -> None:
        if motion is not None:
            motion.advance()  # move the substrate, then re-assemble mass + diffusion on the deformed geometry
            fresh = petsc.assemble_matrix(a_form)
            fresh.assemble()
            ksp.setOperators(fresh)
            matrix_cell[0].destroy()
            matrix_cell[0] = fresh
        rhs = _accumulate_ghosts(petsc.assemble_vector(rhs_local_form))
        coupling_b = _accumulate_ghosts(petsc.assemble_vector(rhs_coupling_form))  # the lagged binding forcing
        rhs.axpy(1.0, coupling_b)
        coupling_b.destroy()
        if rhs_dilution_form is not None:
            dilution_b = _accumulate_ghosts(petsc.assemble_vector(rhs_dilution_form))  # the lagged ALE dilution forcing
            rhs.axpy(1.0, dilution_b)
            dilution_b.destroy()
        ksp.solve(rhs, solution)
        values = solution.array_r
        u_in_fn.x.array[:n_in] = values[:n_in]
        u_out_fn.x.array[:n_out] = values[n_in : n_in + n_out]
        rho_fn.x.array[:n_mem] = values[n_in + n_out : n_in + n_out + n_mem]
        for fn in (u_in_fn, u_out_fn, rho_fn):
            fn.x.scatter_forward()
        for fn, prev in ((u_in_fn, u_in_prev), (u_out_fn, u_out_prev), (rho_fn, rho_prev)):
            prev.x.array[:] = fn.x.array
        rhs.destroy()

    return MembraneCoupledProblem(inner_species, outer_species, membrane_species, u_in_fn, u_out_fn, rho_fn, advance)


class _MatrixFreeShiftedJacobian:
    """PETSc Python-matrix context for the matrix-free action of the TS implicit Jacobian
    ``σ·∂F/∂u̇ + ∂F/∂u``, computed by finite-differencing the residual at the shifted base state.

    This is the crux of the membrane-coupled MOL solve. The *exact* analytic Jacobian cannot be
    assembled: the coupling's ``∂/∂(membrane species)`` block is a bulk-coefficient × membrane-trial term
    on the interface facet, which assembles to **zero** in DOLFINx 0.10 *and* 0.11 (a mixed
    codim-0/codim-1 restriction-assembly limitation — verified directly). Finite-differencing the
    *residual* (which assembles correctly and is exactly mass-conservative) recovers that block, giving
    quadratic Newton convergence; the assembled partial Jacobian preconditions the matrix-free operator.
    `set_base` is called from the TS IJacobian each Newton update."""

    def __init__(self, residual: Callable[[PETSc.Vec, PETSc.Vec, PETSc.Vec], None], template: PETSc.Vec) -> None:
        self._residual = residual  # residual(state, rate, out): assemble F(state, rate) into out
        self._u, self._udot = template.duplicate(), template.duplicate()
        self._sigma = 0.0
        self._f_base = template.duplicate()
        self._u_pert, self._udot_pert, self._f_pert = template.duplicate(), template.duplicate(), template.duplicate()

    def set_base(self, u: PETSc.Vec, udot: PETSc.Vec, sigma: float) -> None:
        u.copy(self._u)
        udot.copy(self._udot)
        self._sigma = sigma
        self._residual(self._u, self._udot, self._f_base)

    def mult(self, _mat: PETSc.Mat, v: PETSc.Vec, y: PETSc.Vec) -> None:
        # Directional derivative of the (shifted) residual: perturb u by εv and u̇ by εσv.
        # petsc4py types Vec.norm() as float | tuple; with no norm_type it returns a float.
        eps = 1.0e-7 * (1.0 + float(self._u.norm())) / (float(v.norm()) + 1.0e-30)  # type: ignore[arg-type]
        self._u.copy(self._u_pert)
        self._u_pert.axpy(eps, v)
        self._udot.copy(self._udot_pert)
        self._udot_pert.axpy(eps * self._sigma, v)
        self._residual(self._u_pert, self._udot_pert, self._f_pert)
        self._f_pert.copy(y)
        y.axpy(-1.0, self._f_base)
        y.scale(1.0 / eps)


def integrate_membrane_coupled(
    md: MathDescription,
    geometry: InterfaceCoupledGeometry,
    *,
    t_final: float,
    dt_initial: float | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    velocity: str | None = None,
    motion: MembraneCoupledMeshMotion | ForceBalanceMeshMotion | None = None,
    motion_steps: int = 10,
    output_times: Sequence[float] = (),
    on_output: Callable[[float, fem.Function, fem.Function, fem.Function], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> MembraneCoupledResult:
    """Integrate a **membrane species coupled to both bulks** to `t_final` with the method-of-lines
    integrator (PETSc TS adaptive BDF) — the fully-implicit counterpart of `assemble_membrane_coupled`'s
    backward-Euler/IMEX step. The binding is implicit (no lagging), so it is unconditionally stable for
    stiff kinetics, and the adaptive time error is ≈0; the same multi-species, symbol-table structure as
    the BE assembler (any number of bulk + surface species).

    **Matrix-free Newton.** The coupling is referenced through the *state* functions, so its Jacobian is
    needed — but the block ``∂(binding)/∂(membrane species)`` (a bulk-coefficient × membrane-trial term on
    `dS`) assembles to zero in DOLFINx 0.10/0.11. So the TS implicit Jacobian is applied **matrix-free**
    (`_MatrixFreeShiftedJacobian`: finite differences of the residual, which assembles fine and is exactly
    conservative), preconditioned by the *assembled partial* Jacobian (`σ·mass + diffusion + the
    assemblable coupling blocks`). This recovers quadratic Newton convergence + round-off conservation.

    Cross-mesh mechanics are the BE assembler's (region `dx`, surface `dx`, interface `dS`, `EntityMap`s,
    `membrane_trace`). The residual splits into a per-mesh local part (mass/rate + diffusion) and an
    interface coupling part (all on the parent `dS`), summed monolithically (`kind="mpi"`).

    **Moving membrane (`velocity` or `motion` given).** The migrating-cell solve: the stiff binding is
    integrated adaptively *while* the membrane deforms. The motion is taken in `motion_steps` discrete outer
    steps — a prescribed `MembraneCoupledMeshMotion` (from `velocity`), or a pre-built `motion=` driver such
    as `ForceBalanceMeshMotion` (the membrane velocity SOLVED from a surface-tension force balance — build
    it with `dt = t_final / motion_steps`, the outer interval). Each outer step moves the substrate, then
    runs the adaptive BDF over that sub-interval on the (frozen) deformed geometry. The **dilution is fully
    implicit** here — unlike the BE/IMEX lagging — because its coefficient is *geometry* (the mesh
    divergence), not a state, so its Jacobian block IS assemblable and goes into both the residual and the
    preconditioner (only the binding stays matrix-free): the residual gains `+ c (∇·v_mesh) w` on the bulk
    regions and the mandatory `+ ρ (∇_Γ·v_Γ) w` on the membrane, with `∇_Γ·v_Γ` the **surface** divergence
    `div(v) − n·(∇v)·n` (n = CellNormal), not `ufl.div` (see `assemble_membrane_coupled`). Total ligand is
    conserved to round-off from the binding cancellation and to O(motion-step) from the outer splitting of
    move-vs-dilute.

    **In-bulk reactions.** A bulk species may carry a `source` (`∂c/∂t = … + source`) referencing any
    sibling species in its own compartment by name. Unlike the BE assembler (affine only), here it may be
    **nonlinear** (e.g. mass-action `A*B`): the residual is evaluated at the state functions so the
    matrix-free Newton differences it exactly, and the assemblable linearisation (`ufl.derivative`) is added
    to the preconditioner.

    **Outputs (fixed geometry).** ``on_output(t, inner, outer, membrane)`` receives the three regions'
    fields — vector P1 ``Function``s, one component per species in declaration order (snapshots valid for
    the call) — at each of ``output_times`` in [0, ``t_final``], recorded from a ``TS`` monitor by
    interpolation so the step sequence is unchanged (``backend/output_times.py``); ``on_progress(t)``
    follows every accepted step. The moving case records only its final state.

    Scope: a *prescribed* membrane velocity; the moving case freezes geometry within each outer interval
    (operator-split). Force-balance velocity is a follow-up.
    """

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    inner_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.inner_subdomain]
    outer_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.outer_subdomain]
    membrane_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.membrane_subdomain]
    if not inner_eqs or not outer_eqs:
        raise NotImplementedError("membrane coupling needs at least one bulk species in each compartment")
    if not membrane_eqs:
        raise NotImplementedError(
            f"membrane coupling needs a surface equation on the membrane subdomain "
            f"{geometry.membrane_subdomain!r} (the species coupled to both bulks)"
        )
    for eq in (*inner_eqs, *outer_eqs):
        if "diffusion" not in eq.terms:
            raise NotImplementedError(f"membrane coupling needs a bulk diffusion equation for {eq.variable!r}")
    inner_species = [eq.variable for eq in inner_eqs]
    outer_species = [eq.variable for eq in outer_eqs]
    membrane_species = [eq.variable for eq in membrane_eqs]

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    membrane_mesh = geometry.membrane_mesh
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1, (len(inner_species),)))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1, (len(outer_species),)))
    V_mem = fem.functionspace(membrane_mesh, ("Lagrange", 1, (len(membrane_species),)))
    u_in_fn, u_out_fn, rho_fn = fem.Function(V_in), fem.Function(V_out), fem.Function(V_mem)
    rate_in, rate_out, rate_mem = fem.Function(V_in), fem.Function(V_out), fem.Function(V_mem)
    n_in = V_in.dofmap.index_map.size_local * V_in.dofmap.index_map_bs
    n_out = V_out.dofmap.index_map.size_local * V_out.dofmap.index_map_bs
    n_mem = V_mem.dofmap.index_map.size_local * V_mem.dofmap.index_map_bs

    mixed = ufl.MixedFunctionSpace(V_in, V_out, V_mem)
    u_in, u_out, rho = ufl.TrialFunctions(mixed)
    w_in, w_out, w_rho = ufl.TestFunctions(mixed)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    dx_mem = ufl.Measure("dx", domain=membrane_mesh)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags, metadata={"quadrature_degree": 4})(
        geometry.interface_tag
    )
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map, geometry.membrane_entity_map]

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    surf_params = _param_symbols(md, membrane_mesh)
    surf_ctx = CompileContext(membrane_mesh, {"geom.x": ufl.SpatialCoordinate(membrane_mesh), **surf_params})

    # Moving membrane (optional): the motion driver + the per-region dilution coefficient `∇·v_mesh`. The
    # mesh moves in `motion_steps` outer steps of length `interval`, so the harmonic displacement carries
    # that dt and `∇·v_mesh = div(disp)/interval`. The membrane uses the SURFACE divergence (see the BE
    # assembler). These coefficients reference the (moving) geometry, so they update as the mesh advances.
    interval = t_final / motion_steps
    if motion is None and velocity is not None:
        motion = MembraneCoupledMeshMotion(md, geometry, velocity=velocity, dt=interval)
    dil_in = dil_out = dil_mem = None
    if motion is not None:
        # The GCL-consistent effective dilution rate per moving mesh, `ln(|Kⁿ⁺¹|/|Kⁿ|)/interval`, from the
        # actual per-cell (bulk) / per-facet (membrane) swept volumes. The TS integrates continuously while
        # the mesh jumps discretely at each outer step, so using this rate (not the instantaneous ∇·v) makes
        # the over-a-stride decay `exp(−d_eff·interval) = |Kⁿ|/|Kⁿ⁺¹|` exactly cancel the jump — each field's
        # substance is conserved (no O(interval) split drift). The membrane facet-area ratio *is* ∫∇_Γ·v_Γ,
        # so no explicit surface-divergence / curvature correction is needed. (Refreshed each `advance`.)
        dil_in = motion.effective_dilution("in")
        dil_out = motion.effective_dilution("out")
        dil_mem = motion.effective_dilution("mem")

    # --- residual: local (per-mesh mass/rate + diffusion + dilution) + coupling (interface dS, implicit) -
    # MOL form: the time derivative ċ = rate (a Function the TS supplies); diffusion uses the state.
    shift = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]  # the TS σ on the bulk meshes
    shift_mem = fem.Constant(membrane_mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # σ on the membrane mesh
    local_terms: list[UflExpr] = []
    precond_terms: list[UflExpr] = []
    for state, rate, trial, test, eqs, dx, region_ctx, sigma, dilution, is_bulk in (
        (u_in_fn, rate_in, u_in, w_in, inner_eqs, dx_in, ctx, shift, dil_in, True),
        (u_out_fn, rate_out, u_out, w_out, outer_eqs, dx_out, ctx, shift, dil_out, True),
        (rho_fn, rate_mem, rho, w_rho, membrane_eqs, dx_mem, surf_ctx, shift_mem, dil_mem, False),
    ):
        # In-bulk reactions reference sibling species in the same compartment via the STATE functions, so a
        # nonlinear source (e.g. mass-action A*B) is fine here — the matrix-free Newton differences the
        # residual; the assemblable linearisation is added to the preconditioner.
        react_ctx = (
            CompileContext(parent, {**ctx.symbols, **{e.variable: state[j] for j, e in enumerate(eqs)}})
            if is_bulk
            else None
        )
        for k, eq in enumerate(eqs):
            _refuse_lab_frame_advection(eq)
            local_terms.append(rate[k] * test[k] * dx)  # ċ·w
            precond_terms.append(sigma * trial[k] * test[k] * dx)  # σ·mass (the preconditioner's ∂/∂rate)
            diffusion = eq.terms.get("diffusion")
            if diffusion is not None:
                d = compile_expression(parse(diffusion), region_ctx)
                local_terms.append(d * ufl.dot(ufl.grad(state[k]), ufl.grad(test[k])) * dx)
                precond_terms.append(d * ufl.dot(ufl.grad(trial[k]), ufl.grad(test[k])) * dx)
            advection = eq.terms.get("relative_advection")
            if advection is not None:  # drift relative to the mesh, `w_rel·∇c` (linear → into residual + Jacobian)
                drift = compile_expression(parse(advection), region_ctx)
                local_terms.append(ufl.dot(drift, ufl.grad(state[k])) * test[k] * dx)
                precond_terms.append(ufl.dot(drift, ufl.grad(trial[k])) * test[k] * dx)
            if dilution is not None:  # implicit ALE dilution: coefficient is geometry, so its Jacobian assembles
                local_terms.append(dilution * state[k] * test[k] * dx)
                precond_terms.append(dilution * trial[k] * test[k] * dx)
            source = eq.terms.get("source")
            if is_bulk and source is not None and react_ctx is not None:  # in-bulk reaction (∂c/∂t = … + source)
                reaction_form = -compile_expression(parse(source), react_ctx) * test[k] * dx
                local_terms.append(reaction_form)
                precond_terms.append(ufl.derivative(reaction_form, state, trial))  # assemblable linearisation
    f_local = _sum_forms(local_terms)
    j_local = _sum_forms(precond_terms)

    # Reservoir Dirichlet on the outer box wall, weakly via a penalty (see the BE assembler). Linear in the
    # outer state, so it goes into both the residual and the assembled Jacobian (the matrix-free part stays
    # the binding only).
    ds_wall = ufl.Measure("ds", domain=parent, subdomain_data=geometry.facet_tags)(geometry.outer_tag)
    for k, g_value in _reservoir_dirichlet(md, geometry, outer_species, ctx):
        f_local += _RESERVOIR_PENALTY * (u_out_fn[k] - g_value) * w_out[k] * ds_wall
        j_local += _RESERVOIR_PENALTY * u_out[k] * w_out[k] * ds_wall

    # Coupling, FULLY IMPLICIT: every species in scope via its interface representation (state functions).
    coupling_symbols: dict[str, UflExpr] = {"geom.x": ufl.SpatialCoordinate(parent), **params}
    for k, name in enumerate(inner_species):
        coupling_symbols[name] = membrane_trace(u_in_fn[k])
    for k, name in enumerate(outer_species):
        coupling_symbols[name] = membrane_trace(u_out_fn[k])
    for k, name in enumerate(membrane_species):
        coupling_symbols[name] = rho_fn[k]("+")
    coupling_ctx = CompileContext(parent, coupling_symbols)
    zero = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    structural = (
        [membrane_trace(w_in[k]) for k in range(len(inner_species))]
        + [membrane_trace(w_out[k]) for k in range(len(outer_species))]
        + [w_rho[k]("+") for k in range(len(membrane_species))]
    )
    f_coup = zero * _sum_forms(structural) * ds_int
    for k, eq in enumerate(membrane_eqs):
        source = eq.terms.get("source")
        if source is not None:
            reaction = compile_expression(parse(source), coupling_ctx)
            f_coup += -reaction * w_rho[k]("+") * ds_int
    bulk_test_of = {name: (w_in, k) for k, name in enumerate(inner_species)}
    bulk_test_of.update({name: (w_out, k) for k, name in enumerate(outer_species)})
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this integrator handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in bulk_test_of:
            continue
        flux = compile_expression(parse(bc.expression), coupling_ctx)
        test, k = bulk_test_of[bc.variable]
        f_coup += -flux * membrane_trace(test[k]) * ds_int
    # The assemblable part of the coupling Jacobian (the bulk-trial columns; the membrane-trial columns
    # are zero — that block is what the matrix-free operator supplies). Used only for preconditioning.
    states, trials = [u_in_fn, u_out_fn, rho_fn], [u_in, u_out, rho]
    j_coup = _sum_forms([ufl.derivative(f_coup, states[j], trials[j]) for j in range(3)])

    f_local_form = fem.form(ufl.extract_blocks(f_local), entity_maps=emaps)
    f_coup_form = fem.form(ufl.extract_blocks(f_coup), entity_maps=emaps)
    j_local_form = fem.form(ufl.extract_blocks(j_local), entity_maps=emaps)
    j_coup_form = fem.form(ufl.extract_blocks(j_coup), entity_maps=emaps)

    for fn, eqs, region_ctx in (
        (u_in_fn, inner_eqs, ctx),
        (u_out_fn, outer_eqs, ctx),
        (rho_fn, membrane_eqs, surf_ctx),
    ):
        _interpolate_component_ics(fn, eqs, region_ctx)

    def unpack(x: PETSc.Vec) -> None:
        u_in_fn.x.array[:n_in] = x.array_r[:n_in]
        u_out_fn.x.array[:n_out] = x.array_r[n_in : n_in + n_out]
        rho_fn.x.array[:n_mem] = x.array_r[n_in + n_out : n_in + n_out + n_mem]
        for fn in (u_in_fn, u_out_fn, rho_fn):
            fn.x.scatter_forward()

    def residual(state: PETSc.Vec, rate: PETSc.Vec, out: PETSc.Vec) -> None:
        unpack(state)
        rate_in.x.array[:n_in] = rate.array_r[:n_in]
        rate_out.x.array[:n_out] = rate.array_r[n_in : n_in + n_out]
        rate_mem.x.array[:n_mem] = rate.array_r[n_in + n_out : n_in + n_out + n_mem]
        for fn in (rate_in, rate_out, rate_mem):  # ghost dofs of ċ too — see integrate_interface_coupled
            fn.x.scatter_forward()
        b = _accumulate_ghosts(petsc.assemble_vector(f_local_form, kind="mpi"))
        coupling_b = _accumulate_ghosts(petsc.assemble_vector(f_coup_form, kind="mpi"))
        b.axpy(1.0, coupling_b)
        b.copy(out)
        b.destroy()
        coupling_b.destroy()

    def evaluate_residual(_ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        residual(x, x_dot, result)

    # Preconditioner holder with the union sparsity of (σ·mass + diffusion) and the assemblable coupling.
    shift.value, shift_mem.value = 1.0, 1.0
    precond = petsc.assemble_matrix(j_local_form, kind="mpi")
    precond.assemble()
    coupling_pre = petsc.assemble_matrix(j_coup_form, kind="mpi")
    coupling_pre.assemble()
    precond.axpy(1.0, coupling_pre, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    coupling_pre.destroy()

    state_vec = petsc.create_vector([V_in, V_out, V_mem], kind="mpi")
    state_vec.array[:n_in] = u_in_fn.x.array[:n_in]
    state_vec.array[n_in : n_in + n_out] = u_out_fn.x.array[:n_out]
    state_vec.array[n_in + n_out : n_in + n_out + n_mem] = rho_fn.x.array[:n_mem]

    mf = _MatrixFreeShiftedJacobian(residual, state_vec)
    sizes = state_vec.getSizes()  # (local, global); petsc4py types it loosely as int | tuple
    operator = PETSc.Mat().createPython((sizes, sizes), comm=parent.comm)  # type: ignore[arg-type]
    operator.setPythonContext(mf)
    operator.setUp()

    def evaluate_jacobian(
        _ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, sigma: float, _mat: PETSc.Mat, pre: PETSc.Mat
    ) -> None:
        mf.set_base(x, x_dot, sigma)  # refresh the matrix-free operator's linearisation point
        unpack(x)
        shift.value, shift_mem.value = sigma, sigma
        fresh = petsc.assemble_matrix(j_local_form, kind="mpi")
        fresh.assemble()
        coupling_fresh = petsc.assemble_matrix(j_coup_form, kind="mpi")
        coupling_fresh.assemble()
        # The coupling adds off-diagonal blocks absent from the local part → its union is `pre`'s sparsity.
        fresh.axpy(1.0, coupling_fresh, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        fresh.copy(pre, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        pre.assemble()
        fresh.destroy()
        coupling_fresh.destroy()

    def make_ts(t0: float, t1: float) -> PETSc.TS:
        ts = PETSc.TS().create(parent.comm)
        ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
        ts.setType("bdf")
        ts.setIFunction(evaluate_residual, state_vec.duplicate())
        ts.setIJacobian(evaluate_jacobian, operator, precond)  # matrix-free operator, assembled preconditioner
        ts.setTime(t0)
        ts.setTimeStep(dt_initial if dt_initial is not None else (t1 - t0) / 1.0e4)
        ts.setMaxTime(t1)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
        ts.setTolerances(atol, rtol)
        ts.setMaxSNESFailures(-1)
        snes = ts.getSNES()
        snes.setUseEW(False)
        snes.getKSP().setType(ksp_type)
        set_preconditioner(snes.getKSP(), pc_type)  # ILU → block-Jacobi/ILU(0) under MPI
        ts.setFromOptions()
        return ts

    if motion is None:
        ts = make_ts(0.0, t_final)
        monitor: OutputMonitor | None = None
        work: PETSc.Vec | None = None
        if on_output is not None or on_progress is not None:
            snaps = (fem.Function(V_in), fem.Function(V_out), fem.Function(V_mem))
            work = state_vec.duplicate()

            def emit(t: float, x: PETSc.Vec) -> None:
                snaps[0].x.array[:n_in] = x.array_r[:n_in]
                snaps[1].x.array[:n_out] = x.array_r[n_in : n_in + n_out]
                snaps[2].x.array[:n_mem] = x.array_r[n_in + n_out : n_in + n_out + n_mem]
                for snap in snaps:
                    snap.x.scatter_forward()
                if on_output is not None:
                    on_output(t, *snaps)

            monitor = OutputMonitor(
                output_times if on_output is not None else (),
                t_start=0.0,
                t_final=t_final,
                work=work,
                emit=emit,
                progress=on_progress,
            )
            ts.setMonitor(monitor)
        ts.solve(state_vec)
        steps, final_time = ts.getStepNumber(), float(ts.getTime())
        if monitor is not None:
            monitor.finish(ts, final_time, state_vec)
        ts.destroy()
        if work is not None:
            work.destroy()
    else:
        if on_output is not None:
            raise NotImplementedError(
                "per-output-time recording of a moving membrane-coupled solve is not implemented; "
                "it records its final state only"
            )
        # Operator-split moving solve: each outer step moves the substrate, then the adaptive BDF integrates
        # the stiff binding over that interval on the (frozen) deformed geometry — a FRESH TS per interval so
        # the BDF history never spans a geometry jump. `state_vec` carries the co-moving field across moves.
        steps = 0
        for i in range(motion_steps):
            motion.advance()
            ts = make_ts(i * interval, (i + 1) * interval)
            ts.solve(state_vec)
            steps += ts.getStepNumber()
            ts.destroy()
        final_time = t_final
    unpack(state_vec)
    for obj in (state_vec, operator, precond):
        obj.destroy()
    return MembraneCoupledResult(
        inner_species, outer_species, membrane_species, u_in_fn, u_out_fn, rho_fn, steps, final_time
    )


def _require_equation(equations: dict[str, TemplateEquation], subdomain: str) -> TemplateEquation:
    eq = equations.get(subdomain)
    if eq is None or "diffusion" not in eq.terms:
        raise NotImplementedError(f"interface coupling needs a bulk diffusion equation on compartment {subdomain!r}")
    return eq


def _interpolate_ic(field: fem.Function, equation: TemplateEquation, ctx: CompileContext) -> None:
    if equation.initial_condition is None:
        return
    mesh = field.function_space.mesh
    local_ctx = CompileContext(
        mesh, {"geom.x": ufl.SpatialCoordinate(mesh), **{k: v for k, v in ctx.symbols.items() if k != "geom.x"}}
    )
    expression = fem.Expression(
        compile_expression(parse(equation.initial_condition), local_ctx),
        field.function_space.element.interpolation_points,
    )
    field.interpolate(expression)


def _interpolate_component_ics(field: fem.Function, equations: list[TemplateEquation], ctx: CompileContext) -> None:
    """Interpolate the equations' initial conditions into the vector `field` (its components are the
    equations' variables, in order; a missing IC defaults to 0). The whole vector is interpolated at
    once via `as_vector` — a one-component space has no `sub(0)` to interpolate into — with the
    expressions recompiled on `field`'s own mesh (reusing the parameter constants from `ctx`)."""
    mesh = field.function_space.mesh
    local_ctx = CompileContext(
        mesh, {"geom.x": ufl.SpatialCoordinate(mesh), **{k: v for k, v in ctx.symbols.items() if k != "geom.x"}}
    )
    components = [
        compile_expression(parse(eq.initial_condition), local_ctx) if eq.initial_condition is not None else ufl.zero()
        for eq in equations
    ]
    field.interpolate(fem.Expression(ufl.as_vector(components), field.function_space.element.interpolation_points))


def _refuse_lab_frame_advection(eq: Any) -> None:
    """The lab-frame ``advection`` slot is assembled by the single-mesh path only (``assemble``)."""

    if "advection" in eq.terms:
        raise NotImplementedError(
            f"the lab-frame 'advection' slot (equation for {eq.variable!r}) is supported on the single-mesh "
            "path only; the interface-coupled solvers take 'relative_advection'"
        )
