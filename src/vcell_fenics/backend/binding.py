"""Mixed-dimensional reference solver for §1.6.6 — ligand-receptor binding.

This is the first cross-mesh coupled solve in the backend, the consumer of the
multi-compartment geometry and the `trace(·)` operator. The §1.6.6 model: an
extracellular ligand L diffuses in the bulk Ω_ext (an annulus), binds reversibly to
the membrane-bound free receptor ρ_f to form the bound complex ρ_b, with the outer
boundary held at a fixed reservoir concentration (Dirichlet). The coupling is
bulk↔surface — the binding rate `b = k_on·trace(L)·ρ_f − k_off·ρ_b` enters L's
Neumann flux at the membrane *and* the surface sources for ρ_f, ρ_b.

**Why this is a dedicated solver, not the formalism `assemble()` path.** L lives on
the bulk mesh and ρ_f, ρ_b on the membrane submesh, so the system is a *block*
problem over two meshes — outside the single-mesh `DiscreteProblem` IR. Generalising
the mixed-dimensional block assembly into the formalism (auto-detecting cross-subdomain
coupling from arbitrary expressions, a block lowering) is a larger increment; this
module proves the mechanics end-to-end on the canonical model, driven by its
parameters (`BindingParameters.from_math_description`).

**Mixed-dimensional assembly mechanics** (validated, and the reusable knowledge):

- All coupling is assembled over the **bulk's membrane facets** (`ds`), where L is
  native and the surface functions ρ_f, ρ_b (and test functions) are pulled in via
  the DOLFINx 0.10 `entity_maps=[EntityMap]` argument to `fem.form`. The reverse —
  integrating over the membrane submesh while referencing the bulk L (a true "trace"
  integral) — is *not* supported by FFC (it cannot tabulate a 2D element on a 1D
  cell), so the bulk-`ds` route is the only one.
- A single UFL form cannot mix two integration meshes, so the surface terms (mass +
  surface diffusion on the membrane `dx`) and the coupling terms (on the bulk `ds`)
  are assembled as two block matrices and summed (`A = A_local + A_coupling`).
- The bilinear binding `k_on·trace(L)·ρ_f` is linearised **semi-implicitly by lagging
  the slowly-varying bulk L** (`b = k_on·Lⁿ·ρ_f − k_off·ρ_b`): the fast membrane
  reaction stays implicit (unconditionally stable, ρ_f never goes negative), and the
  coupling matrix re-assembles each step as Lⁿ changes. Because L is lagged it never
  appears as a trial in the coupling block, leaving an all-`None` column that
  `create_matrix` cannot deduce a space for — a structural-zero `0·L·w_L ds` entry
  restores the column at no cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.approaches.multicompartment.geometry import (
    MEMBRANE_TAG,
    OUTER_TAG,
    ExtracellularAnnulus,
)
from vcell_fenics.formalism.schema import MathDescription, ParameterConstant, TemplateEquation


@dataclass(frozen=True)
class BindingParameters:
    """The §1.6.6 rate / transport constants."""

    k_on: float
    k_off: float
    l_reservoir: float
    d_ligand: float
    d_surface: float

    @classmethod
    def from_math_description(cls, md: MathDescription) -> BindingParameters:
        """Pull the §1.6.6 constants out of a MathDescription: the rate parameters
        from `parameters`, and the diffusion coefficients from the ligand (bulk) and
        a receptor (surface) equation's `diffusion` slot."""

        params = {p.name: p.value for p in md.parameters if isinstance(p, ParameterConstant)}
        equations = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
        bulk = next(eq for eq in equations if eq.template == "bulk_radv_diff")
        surface = next(eq for eq in equations if eq.template == "surface_pde_with_dilution")
        return cls(
            k_on=params["k_on"],
            k_off=params["k_off"],
            l_reservoir=params["L_reservoir"],
            d_ligand=float(bulk.terms["diffusion"]),
            d_surface=float(surface.terms["diffusion"]),
        )


@dataclass
class BindingState:
    """The three coupled fields: ligand on the bulk, free / bound receptor on the
    membrane."""

    ligand: fem.Function
    free_receptor: fem.Function
    bound_receptor: fem.Function


class LigandReceptorBinding:
    """The §1.6.6 coupled solve, advanced by backward-Euler steps.

    Construct with the annulus geometry, the parameters, the step `dt`, and the
    initial values; call `step()` to advance and read `state` / `receptor_total()`.
    """

    def __init__(
        self,
        annulus: ExtracellularAnnulus,
        params: BindingParameters,
        *,
        dt: float,
        ligand_ic: float = 1.0,
        free_ic: float = 0.5,
        bound_ic: float = 0.0,
    ) -> None:
        self._annulus = annulus
        self._params = params
        bulk = annulus.bulk_mesh
        membrane = annulus.membrane_mesh
        emaps = [annulus.membrane_entity_map]
        tdim = bulk.topology.dim

        V_l = fem.functionspace(bulk, ("Lagrange", 1))
        V_f = fem.functionspace(membrane, ("Lagrange", 1))
        V_b = fem.functionspace(membrane, ("Lagrange", 1))
        self._n_ligand = V_l.dofmap.index_map.size_local
        self._n_surface = V_f.dofmap.index_map.size_local

        self._ligand, self._ligand_prev = fem.Function(V_l, name="L"), fem.Function(V_l)
        self._free, self._free_prev = fem.Function(V_f, name="rho_f"), fem.Function(V_f)
        self._bound, self._bound_prev = fem.Function(V_b, name="rho_b"), fem.Function(V_b)
        for field, value in ((self._ligand, ligand_ic), (self._free, free_ic), (self._bound, bound_ic)):
            field.x.array[:] = value
        self._commit()  # previous := current

        dx = ufl.Measure("dx", domain=bulk)
        dx_s = ufl.Measure("dx", domain=membrane)
        ds_mem = ufl.Measure("ds", domain=bulk, subdomain_data=annulus.facet_tags)(MEMBRANE_TAG)
        u_l, w_l = ufl.TrialFunction(V_l), ufl.TestFunction(V_l)
        u_f, w_f = ufl.TrialFunction(V_f), ufl.TestFunction(V_f)
        u_b, w_b = ufl.TrialFunction(V_b), ufl.TestFunction(V_b)
        self._dt = dt

        # Local (single-mesh) blocks: backward-Euler mass + diffusion, on each field's
        # own mesh. The ligand diffuses in the bulk; the receptors on the membrane.
        # `dt` and the diffusion coefficients are plain Python scalars — a
        # `fem.Constant` carries a mesh, and a bulk constant in a membrane integral
        # trips ffcx's cross-mesh tabulation (the same limitation as a trace integral).
        a_local = [
            [u_l * w_l * dx + dt * params.d_ligand * ufl.inner(ufl.grad(u_l), ufl.grad(w_l)) * dx, None, None],
            [None, u_f * w_f * dx_s + dt * params.d_surface * ufl.inner(ufl.grad(u_f), ufl.grad(w_f)) * dx_s, None],
            [None, None, u_b * w_b * dx_s + dt * params.d_surface * ufl.inner(ufl.grad(u_b), ufl.grad(w_b)) * dx_s],
        ]
        self._a_local = petsc.assemble_matrix(fem.form(a_local, entity_maps=emaps))
        self._a_local.assemble()

        # Coupling blocks on the bulk membrane facets. b = k_on·Lⁿ·ρ_f − k_off·ρ_b
        # (L lagged). Residual contributions: −dt·b·w_l, +dt·b·w_f, −dt·b·w_b. The
        # structural zero in the (L,L) slot keeps the (lagged) ligand column present.
        k_on, k_off = params.k_on, params.k_off
        lp = self._ligand_prev
        zero = fem.Constant(bulk, PETSc.ScalarType(0.0))  # type: ignore[operator]  # bulk-ds form, bulk constant is fine
        a_coupling = [
            [zero * u_l * w_l * ds_mem, -dt * k_on * lp * u_f * w_l * ds_mem, dt * k_off * u_b * w_l * ds_mem],
            [None, dt * k_on * lp * u_f * w_f * ds_mem, -dt * k_off * u_b * w_f * ds_mem],
            [None, -dt * k_on * lp * u_f * w_b * ds_mem, dt * k_off * u_b * w_b * ds_mem],
        ]
        self._coupling_form = fem.form(a_coupling, entity_maps=emaps)

        # Backward-Euler RHS: the previous step's values (mass term), per mesh.
        self._rhs_form = fem.form(
            [self._ligand_prev * w_l * dx, self._free_prev * w_f * dx_s, self._bound_prev * w_b * dx_s],
            entity_maps=emaps,
        )

        # Reservoir Dirichlet L = L_reservoir on the outer boundary.
        bulk.topology.create_connectivity(tdim - 1, tdim)
        self._ligand_outer_dofs = fem.locate_dofs_topological(V_l, tdim - 1, annulus.facet_tags.find(OUTER_TAG)).astype(
            np.int32
        )

    def step(self) -> None:
        """Advance one backward-Euler step."""

        # A = A_local + A_coupling(Lⁿ); the coupling re-assembles as Lⁿ changes.
        coupling = petsc.assemble_matrix(self._coupling_form)
        coupling.assemble()
        matrix = self._a_local.copy()
        matrix.axpy(1.0, coupling, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        matrix.zeroRows(self._ligand_outer_dofs, diag=1.0)  # Dirichlet rows (ligand block, offset 0)

        rhs = petsc.assemble_vector(self._rhs_form)
        rhs.array[self._ligand_outer_dofs] = self._params.l_reservoir

        solution = matrix.createVecRight()
        ksp = PETSc.KSP().create(self._annulus.bulk_mesh.comm)
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(rhs, solution)

        values = solution.array_r
        n_l, n_s = self._n_ligand, self._n_surface
        self._ligand.x.array[:n_l] = values[:n_l]
        self._free.x.array[:n_s] = values[n_l : n_l + n_s]
        self._bound.x.array[:n_s] = values[n_l + n_s : n_l + 2 * n_s]
        self._commit()
        ksp.destroy()
        coupling.destroy()
        matrix.destroy()
        solution.destroy()

    def _commit(self) -> None:
        self._ligand_prev.x.array[:] = self._ligand.x.array
        self._free_prev.x.array[:] = self._free.x.array
        self._bound_prev.x.array[:] = self._bound.x.array

    @property
    def state(self) -> BindingState:
        return BindingState(self._ligand, self._free, self._bound)

    def receptor_total(self) -> float:
        """∫_Γ (ρ_f + ρ_b) ds — the conserved total receptor on the membrane."""

        membrane = self._free.function_space.mesh
        form = fem.form((self._free + self._bound) * ufl.dx(domain=membrane))
        return float(membrane.comm.allreduce(fem.assemble_scalar(form).real, op=MPI.SUM))
