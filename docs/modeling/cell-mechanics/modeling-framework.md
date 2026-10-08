# Cell kinematics and mechanics: modeling framework

**Status:** initial proposal for discussion, 2026-10-08. This records the next modeling work, not a
complete constitutive theory or an implementation specification. The earlier “Constitutive laws
overview” discussion has not been fully recovered; its unavailable equations and examples are not
reconstructed here. [Workspace guide](README.md) ·
[Documentation discrepancies](../../reviews/2026-10-08-documentation-discrepancies.md).

This iteration focuses on the new continuum physics: kinematics, bulk/surface balances, phase
coupling, constitutive laws, interfaces and verification. Earlier ideas about cytoskeletal and motor
mechanics, 0D fiduciaries or vesicle/adhesion domains, and deformable-atlas distance/vector metrics
are recorded as deferred context in the [workspace guide](README.md#deferred-context-from-earlier-modeling-discussions)
and are outside the first model family's scope.

## 1. Modeling goals before solver choices

Develop examples that distinguish the following biological questions:

- How do imposed shape changes redistribute cytosolic and membrane-bound species?
- How do membrane tension and cytoplasmic stresses determine shape and relaxation?
- When do cytoskeletal network and solvent move together, slip or exchange material?
- Which observations require elastic memory, stress relaxation, turnover or remodeling?
- What generates directed cell migration: active stress, polymerization, substrate interactions,
  chemical polarity, or their coupling?

For each example specify measurable outputs: shape and centroid, phase velocities, pressure or
traction, species profiles and totals, and relevant relaxation or migration time scales. State whether
2D is a planar material model, a cross-section, or a thickness-averaged approximation; the parameter
units and conserved measures depend on that choice.

## 2. Kinematic vocabulary and balance laws

Use distinct symbols and declarations for:

| Quantity | Modeling role |
|---|---|
| Current domain Ω(t) and membrane Γ(t) | Where the physical fields live |
| Boundary velocity v_b | How the interface position changes; its normal speed determines shape evolution |
| Material-phase velocity v_a | Motion of network, solvent or membrane material |
| Species carrier velocity v_c | Which phase carries a species, possibly with additional relative transport |
| Mesh velocity w | Numerical coordinates in an ALE discretization |
| Reference/natural configuration and internal state | What a material remembers, if its constitutive law requires memory |

Physical tangential slip and tangential node redistribution are distinct. Normal matching is a physical
interface assumption for impermeable material boundaries; permeation or phase conversion requires a
flux balance instead. Do not impose every phase's velocity equal to boundary motion by default.

For a bulk concentration c measured per current volume, a useful starting conservation law is

```text
∂t c + div(c v_c + j) = r,
```

where j is non-advective flux and r is production per current volume. Its ALE expression is

```text
∂t c|mesh + (v_c − w)·grad(c) + c div(v_c) + div(j) = r.
```

A boundary moving at v_b sees relative outward flux `(c (v_c − v_b) + j)·n`. Declaring a boundary
impermeable therefore differs from setting only the diffusive flux to zero. For concentrations per
phase volume, volume fractions enter storage and flux; that is a separate model choice requiring its
own balance derivation.

For a surface density ρ per current area, with material surface velocity v_Γ and tangential relative
flux j_Γ, the starting balance is

```text
D_t^Γ ρ + ρ div_Γ(v_Γ) + div_Γ(j_Γ) = r_Γ + exchange.
```

Here D_t^Γ follows membrane material, not arbitrary mesh nodes. Derive the mesh-relative expression
when discretizing. Bulk loss and membrane gain must share compatible signs, units and measures.

These are continuous balances; they do not prescribe a time integrator. The existing formalism and FSI
transport paths use different discrete treatments, documented in
[ADR 009](../../decisions/009-fsi-species-transport-in-formalism.md).

## 3. Mechanics: balances, constitutive laws and interface closure

Organize each model in four parts:

1. **Balances:** species/phase mass and momentum, with explicit assumptions about inertia and
   compressibility. Mixture incompressibility is not automatically incompressibility of each phase.
2. **Constitutive laws:** passive viscous or elastic stress, active stress, interphase drag, diffusion
   and any internal-state evolution. State admissible coefficients, units and applicable regimes.
3. **Interface laws:** traction balance, tension/bending if included, normal flux, tangential slip or
   adhesion, and membrane reactions. Distinguish the physical law from its numerical enforcement.
4. **Initial and boundary data:** include reference state, pressure gauge or other null-mode treatment,
   compatibility conditions and external mechanical constraints.

A constitutive law is not always an algebraic stress substitution. Elasticity and viscoelasticity may
introduce deformation gradients, objective history evolution, growth/turnover and an evolving natural
configuration. Remeshing must preserve or consistently transfer that state. The earlier suggestion of
a “poroelastic swap” is a direction to investigate, not evidence that these requirements already exist.

## 4. Candidate mathematical problem families

| Family | Mathematical structure | Repository starting point and remaining modeling work |
|---|---|---|
| Prescribed kinematics with species | Moving-domain transport PDEs; coupled reactions | ALE and transport paths exist; make carrier, boundary flux and conserved measure explicit |
| Passive viscous shape relaxation | Quasi-static constrained momentum coupled to moving geometry and transport | Stokes and surface-tension FSI drivers exist; document closure, null modes and conservation envelope |
| Active viscous mixture | Coupled phase momentum, mixture constraint, phase/species transport | Two-phase drag/FSI foundations exist; define volume fractions, active stress and phase boundary conditions |
| Viscoelastic or poroelastic cytoplasm | Momentum plus elastic/history state and fluid/mixture constraints | A proposed extension; choose reference/state evolution and verification before promising a constitutive plug-in |
| Actin–myosin migration | Compressible active-gel force balance, myosin transport and polymerizing free boundary | Planned in the [migration note](../active-protrusion-migration.md); substrate coupling and symmetry breaking distinguish it from passive relaxation |

“Exists” here means source-inspected infrastructure, not full formalism or CLI support and not a new
numerical validation. The family boundaries are provisional; worked biological examples should decide
which abstractions deserve first-class support.

## 5. A verification ladder for the examples

Use simple discriminating cases before coupled biological examples. Reuse tests and harnesses rather
than duplicating them in a new framework.

| Example | What it distinguishes | Evidence to reuse or produce |
|---|---|---|
| Static diffusion and prescribed expansion | Transport accuracy versus dilution and geometric conservation | Existing eigenmodes, moving/static MMS twins and mass checks |
| Different mesh and carrier velocities | Numerical frame motion versus physical transport | Existing rotating-mesh/lab-frame tests; include a compressible-carrier case |
| Tense circle/sphere and perturbed shape | Mechanical equilibrium versus relaxation | Existing Laplace and ellipse FSI tests; derive dimension-specific expectations |
| Two-phase drag | Independent flow versus phase locking | Existing manufactured mixture and drag tests; examine boundary closure and fraction weighting |
| Elastic/poroelastic relaxation | Stored energy, history and fluid-network coupling | Proposed benchmark such as Biot consolidation, after assumptions and BCs are fixed |
| Active migration | Shape motion or membrane circulation versus centroid translation | Proposed speed, shape and myosin-profile comparison with mbsolver under matched conventions |

For each new case record equations, assumptions, parameters/units, initial and boundary data, expected
invariants, error norms, h/dt refinement and a negative control where useful. Mass conservation alone
cannot establish solution accuracy. Cross-solver agreement is evidence of consistency, not an exact
solution; use analytical or manufactured checks where possible.

Do not carry claims such as “exact conservation” between solver paths. State which invariant, whether
it is continuous or discrete, what tolerance/order is expected, and how remeshing changes the evidence.

## 6. Modeling representation and transformation into the formalism

The biological modeling layer describes concepts such as materials, species, constitutive behavior,
interfaces and interactions. Its transformation derives a closed mathematical problem from those
choices. This is distinct from the solver's translation of that math into UFL and a discrete problem.

The expected production home of the biological representation and model-to-math transformation is
`virtualcell/vcell`, following the current architecture (user clarification, 2026-10-08). This workspace
centralizes the design and can host prototypes of both layers while the concepts are being developed;
it does not propose moving permanent ownership of VCell's modeling layer into the solver repository.
See the [ownership guidance](README.md#repository-ownership-and-the-purpose-of-this-workspace).

For a candidate representation, record a biological input example, assumptions supplied by the user,
the equations/constraints and interface conditions generated from it, and unresolved choices that
must be diagnosed rather than silently defaulted. A prototype here can make that transformation
concrete before the production representation or implementation language is settled.

After a family and its verification are reviewed, map the generated mathematical problem to:

- named compartments and interfaces in `GeometryDescription`;
- variables, equations, parameters and motion in `MathDescription`;
- numerical choices in `SolverConfiguration`;
- dedicated backend drivers where no declarative route exists yet.

Make gaps explicit: multiple physical velocities, phase-weighted storage, reference/history tensors,
interface mechanics, and coupled quasi-static/evolving systems may require new semantics or templates.
Use semantic template names while the [T5 naming collision](../../reviews/2026-10-08-documentation-discrepancies.md#d03--t5-has-two-meanings-in-the-same-formalism)
is unresolved. Promoting a working driver into a supported template needs a defined problem envelope,
validation rules, diagnostics and conformance examples.

A transformation test should check the generated mathematical structure independently of a numerical
solve; a companion conformance example should check the solution. This gives eventual VCell integration
both a model-to-math contract and solver-independent expected behavior, while discretization choices
remain in the backend/configuration layer.

## 7. First decisions for review

- Choose the first biological example: passive shape relaxation, active viscous mixture, or active-gel migration.
- Define concentration measures and boundary permeability for that example.
- Decide which phase fractions and velocities are prescribed versus solved.
- Identify the simplest constitutive law that can answer the biological question, and whether it needs memory.
- Identify the existing entry point and missing capabilities, then select acceptance benchmarks.
- Decide whether a local representation/transformation prototype would resolve an open modeling question;
  define its input/output contract and eventual VCell integration boundary before expanding it.
- Keep deferred point-domain, motor/adhesion and atlas-metric ideas out of the first physics milestone
  unless a concrete example demonstrates that they are required for closure.

Record decisions here with date, rationale and review PR as they are accepted. No choice in this list is
settled by the initial proposal.
