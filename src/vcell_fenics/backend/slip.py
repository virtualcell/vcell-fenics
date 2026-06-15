"""Normal-slip (perfect-slip) velocity boundary condition via Nitsche's method.

The first machinery piece of the multiphase-cytoplasm path
(`docs/modeling/multiphase-cytoplasm-ale.md`, step 1): a velocity boundary condition
that pins only the **normal** component, `v·n = g`, and leaves the tangential traction
**free** (the fluid slips along the boundary). This is the physical interface condition
between a fluid phase and the membrane — *no penetration* (`g = 0`) or *normal-matching*
to a moving boundary (`g = w·n`) — and the building block the eventual fluid momentum
balance needs. It is also the concrete answer to "BCs for the velocity": the normal
component is fundamental (the phases must stay conforming), the tangential is the slip
condition.

Because the boundary normal is not axis-aligned, a pointwise Dirichlet on `v·n` is awkward
in DOLFINx; Nitsche's method enforces it **weakly** instead. For the viscous operator
`-ν∇²v` (+ lower-order terms) the symmetric Nitsche terms over a boundary Γ are

    a_Γ = ∫_Γ [ -ν (n·∇v·n)(w·n) - ν (n·∇w·n)(v·n) + (βν/h)(v·n)(w·n) ] ds
    L_Γ = ∫_Γ [ -ν (n·∇w·n) g                       + (βν/h) g (w·n)   ] ds

added to the momentum weak form. Nothing is imposed on the tangential traction, so it
stays in its natural zero-traction (free-slip) state. The **discrete facet normal** is
used for both `v·n` and `g`, so a no-penetration (`g = 0`) or mesh-velocity-matching
(`g = w·n` built with the same `n`) condition is geometry-consistent on a curved boundary
— building `g` from a *smooth* normal instead would inject an O(h) penalty inconsistency.

`β` is the dimensionless Nitsche penalty (≳ a few); the consistency terms make the scheme
accurate (not merely penalty-limited), so a polynomial solution is recovered to round-off.
The adjoint-term sign is selectable (`symmetric=`): the symmetric variant is L2-optimal
but needs `β` above a threshold for coercivity; the **non-symmetric** variant is coercive
for any `β ≥ 0`, so it runs penalty-free with no parameter to tune — the robust choice for
cut/embedded interfaces or weakly-coercive (screening-free) operators. See
`nitsche_normal_slip`.

Scope: the consistency terms above are for the plain vector-Laplacian traction `ν ∇v·n`.
A symmetric-gradient (Stokes) operator `ν(∇v + ∇vᵀ)` or a pressure term changes the
traction and hence the consistency term — a later extension when the momentum solve lands.
"""

from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh

from vcell_fenics.backend._typing import UflExpr


def nitsche_normal_slip(
    v: UflExpr,
    test: UflExpr,
    normal_velocity: UflExpr,
    *,
    mesh: Mesh,
    ds: ufl.Measure,
    viscosity: UflExpr | float = 1.0,
    beta: float = 20.0,
    symmetric: bool = True,
) -> tuple[UflExpr, UflExpr]:
    """Nitsche terms enforcing `v·n = normal_velocity` weakly with free tangential
    traction (perfect slip), for the `-ν∇²v` viscous operator.

    Returns `(a_terms, L_terms)` — UFL forms over `ds` to **add** to the momentum weak
    form's bilinear and linear parts. `normal_velocity` is the prescribed `v·n` (a scalar
    UFL expression): `0` for no penetration, or `ufl.dot(w_mesh, n)` to match a moving
    boundary. Use the boundary measure `ds` restricted to the interface where slip
    applies.

    `symmetric` selects the adjoint-term sign (θ = −1 vs +1):

    - `True` (default) — **symmetric** Nitsche: adjoint-consistent, so optimal in *L2*
      as well as the energy norm, and the matrix stays symmetric. Needs `beta` above a
      mesh/operator-dependent threshold for coercivity.
    - `False` — **non-symmetric** Nitsche: coercive for *any* `beta ≥ 0`, so it runs
      penalty-free (`beta=0`) with **no stabilisation parameter to tune** — robust where
      the symmetric threshold is fragile (cut/embedded interfaces, weak coercivity, wide
      viscosity contrast). The cost is a non-symmetric matrix (irrelevant under a direct
      or GMRES solve) and possibly half-order-suboptimal L2 (the energy/derivative norm
      stays optimal). Prefer this once the slip lands on cut cells or a screening-free
      Stokes phase; the default suits the current drag-coercive solves.
    """

    theta = -1.0 if symmetric else 1.0
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    nu = viscosity
    dvn = ufl.dot(ufl.dot(ufl.grad(v), n), n)  # n·∇v·n — the normal traction direction
    dtn = ufl.dot(ufl.dot(ufl.grad(test), n), n)
    vn, tn = ufl.dot(v, n), ufl.dot(test, n)
    a = (-nu * dvn * tn + theta * nu * dtn * vn + (beta * nu / h) * vn * tn) * ds
    rhs = (theta * nu * dtn * normal_velocity + (beta * nu / h) * normal_velocity * tn) * ds
    return a, rhs


def solve_overdamped_slip(
    mesh: Mesh,
    *,
    forcing: UflExpr,
    normal_velocity: UflExpr,
    viscosity: float = 1.0,
    screening: float = 1.0,
    beta: float = 20.0,
    symmetric: bool = True,
) -> fem.Function:
    """Solve the overdamped vector field `-ν∇²v + γv = f` on `mesh` with a Nitsche
    normal-slip BC `v·n = normal_velocity` (free tangential) on the whole boundary.

    `forcing` and `normal_velocity` are UFL expressions on `mesh` (build them with
    `ufl.SpatialCoordinate(mesh)` / `ufl.FacetNormal(mesh)`). A reference solver for the
    slip machinery — the screened vector Laplacian stands in for an inertia-free fluid
    momentum balance until the real Stokes/Brinkman operator and pressure land.
    `symmetric` selects the Nitsche variant (see `nitsche_normal_slip`); the GMRES-capable
    direct solve here is agnostic to the resulting (a)symmetry.
    """

    space = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    v, w = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = (viscosity * ufl.inner(ufl.grad(v), ufl.grad(w)) + screening * ufl.inner(v, w)) * ufl.dx
    rhs = ufl.inner(forcing, w) * ufl.dx
    a_bc, rhs_bc = nitsche_normal_slip(
        v, w, normal_velocity, mesh=mesh, ds=ufl.ds(domain=mesh), viscosity=viscosity, beta=beta, symmetric=symmetric
    )
    solution = fem.Function(space, name="velocity")
    LinearProblem(
        a + a_bc,
        rhs + rhs_bc,
        u=solution,
        petsc_options_prefix=f"vcellfenics_slip_{id(solution):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
    return solution
