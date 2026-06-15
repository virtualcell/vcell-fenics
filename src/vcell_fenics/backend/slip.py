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
) -> tuple[UflExpr, UflExpr]:
    """Nitsche terms enforcing `v·n = normal_velocity` weakly with free tangential
    traction (perfect slip), for the `-ν∇²v` viscous operator.

    Returns `(a_terms, L_terms)` — UFL forms over `ds` to **add** to the momentum weak
    form's bilinear and linear parts. `normal_velocity` is the prescribed `v·n` (a scalar
    UFL expression): `0` for no penetration, or `ufl.dot(w_mesh, n)` to match a moving
    boundary. Use the boundary measure `ds` restricted to the interface where slip
    applies.
    """

    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)
    nu = viscosity
    dvn = ufl.dot(ufl.dot(ufl.grad(v), n), n)  # n·∇v·n — the normal traction direction
    dtn = ufl.dot(ufl.dot(ufl.grad(test), n), n)
    vn, tn = ufl.dot(v, n), ufl.dot(test, n)
    a = (-nu * dvn * tn - nu * dtn * vn + (beta * nu / h) * vn * tn) * ds
    rhs = (-nu * dtn * normal_velocity + (beta * nu / h) * normal_velocity * tn) * ds
    return a, rhs


def solve_overdamped_slip(
    mesh: Mesh,
    *,
    forcing: UflExpr,
    normal_velocity: UflExpr,
    viscosity: float = 1.0,
    screening: float = 1.0,
    beta: float = 20.0,
) -> fem.Function:
    """Solve the overdamped vector field `-ν∇²v + γv = f` on `mesh` with a Nitsche
    normal-slip BC `v·n = normal_velocity` (free tangential) on the whole boundary.

    `forcing` and `normal_velocity` are UFL expressions on `mesh` (build them with
    `ufl.SpatialCoordinate(mesh)` / `ufl.FacetNormal(mesh)`). A reference solver for the
    slip machinery — the screened vector Laplacian stands in for an inertia-free fluid
    momentum balance until the real Stokes/Brinkman operator and pressure land.
    """

    space = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    v, w = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = (viscosity * ufl.inner(ufl.grad(v), ufl.grad(w)) + screening * ufl.inner(v, w)) * ufl.dx
    rhs = ufl.inner(forcing, w) * ufl.dx
    a_bc, rhs_bc = nitsche_normal_slip(
        v, w, normal_velocity, mesh=mesh, ds=ufl.ds(domain=mesh), viscosity=viscosity, beta=beta
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
