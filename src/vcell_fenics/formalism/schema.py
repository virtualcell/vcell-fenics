"""In-memory dataclass schema for the declarative formalism.

Mirrors the v1 schema specified in docs/modeling/declarative-formalism.md Part 2
(§2.1 envelope, §2.2 per-entity schema). Surface syntax for expressions is left
as a raw string field; the parser (string → typed AST) and the validator
(§2.5 nine-step pass) are separate modules that consume these dataclasses.

All dataclasses are frozen + kw_only + slotted: the schema is an immutable
data structure, construction is by keyword for clarity, and slots eliminate
per-instance __dict__ overhead. Discriminated unions (Motion, Parameter,
Equation, BoundaryCondition) are tagged unions: one dataclass per `kind`
with a Literal discriminator, unioned by `|` and exported as a TypeAlias.
mypy narrows through `isinstance` and pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

# ---------------------------------------------------------------------------
# Closed enumerations used in multiple places.
# Open identifiers (template names, FE space hints) stay `str`; the validator
# checks them against the live template / space registries.
# ---------------------------------------------------------------------------

SubdomainKind: TypeAlias = Literal["volume", "surface", "curve", "point"]
VariableType: TypeAlias = Literal["scalar", "vector", "symmetric_tensor"]
Temporality: TypeAlias = Literal["time_dependent", "steady_state"]


# ---------------------------------------------------------------------------
# Motion (§1.10, §2.2.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionNone:
    """Static subdomain. Substrate velocity is zero everywhere."""

    kind: Literal["none"] = "none"


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionPrescribedVelocity:
    """Prescribed motion via a velocity field expression. Vector-valued in R^d."""

    kind: Literal["prescribed"] = "prescribed"
    velocity: str


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionPrescribedDisplacement:
    """Prescribed motion via a displacement field expression. Vector-valued in R^d."""

    kind: Literal["prescribed"] = "prescribed"
    displacement: str


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionUnknown:
    """Motion solved by an equation elsewhere in the MathDescription.

    `variable` names a vector-typed Variable defined on the same subdomain;
    some Equation in the MathDescription must govern that variable.
    """

    kind: Literal["unknown"] = "unknown"
    variable: str


Motion: TypeAlias = MotionNone | MotionPrescribedVelocity | MotionPrescribedDisplacement | MotionUnknown


# ---------------------------------------------------------------------------
# Subdomain (§2.2.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class Subdomain:
    """A named topological entity (a *class*, not a region) in the geometry.

    See §1.2.2 for the class-vs-region distinction. `motion` defaults to a
    static subdomain (`MotionNone`); per memory decision 6 motion is a
    property of the subdomain, not of any equation on it.
    """

    name: str
    kind: SubdomainKind
    motion: Motion = field(default_factory=MotionNone)


# ---------------------------------------------------------------------------
# Variable (§2.2.2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class Variable:
    """A named unknown function on exactly one subdomain class.

    `(name, subdomain)` is the unique identifier — the same `name` on a
    different subdomain is a different variable (§1.3.1).
    """

    name: str
    subdomain: str
    type: VariableType = "scalar"
    space: str = "lagrange_p1"


# ---------------------------------------------------------------------------
# Parameter (§2.2.3) — three forms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterConstant:
    """Constant scalar parameter — the common case."""

    kind: Literal["scalar"] = "scalar"
    name: str
    value: float


@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterExpression:
    """Expression-valued parameter (§2.2.3 (b)).

    `expression` is a raw string in the §1.8 vocabulary, evaluated in the
    surrounding context at every use site. `subdomain` is required if the
    expression body references any geometric helper (n, H, theta, …); the
    validator enforces this (§1.11.10) and rejects uses from incompatible
    contexts.
    """

    kind: Literal["expression"] = "expression"
    name: str
    type: VariableType = "scalar"
    expression: str
    subdomain: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterRegionMap:
    """Tier-1 region variation (§1.2.5, §2.2.3 (c)).

    Provides one constant value per region of the named subdomain class. The
    validator (§1.11.4) requires every region of `subdomain` to appear in
    `values`; missing entries are errors, not silent defaults. v1 carries
    per-region *constants* only — per-region expressions are v2 (Appendix B).
    """

    kind: Literal["region_map"] = "region_map"
    name: str
    subdomain: str
    values: dict[str, float]


Parameter: TypeAlias = ParameterConstant | ParameterExpression | ParameterRegionMap


# ---------------------------------------------------------------------------
# Equation (§2.2.4 template form, §2.2.5 weak-form).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class TemplateEquation:
    """An equation instantiated from a named operator template (§1.4).

    `template` is the template's registered name (e.g. `bulk_radv_diff`,
    `surface_pde_with_dilution`). v1 template set: T1–T4 (T5–T7 are v2).
    The template is intentionally typed as `str`: extending the registry in
    v2 should not require a schema change. Unknown names are caught by the
    validator.

    `terms` maps the template's slot names to raw expression strings. Slots
    the template considers optional may be omitted; the validator (§1.11.4,
    §1.11.5) checks per-template slot coverage and types.

    `initial_condition` is required iff `temporality = time_dependent`
    (§1.7.1, §1.9.5). The dataclass keeps it optional; the validator
    enforces the iff-rule structurally.
    """

    template: str
    variable: str
    subdomain: str
    temporality: Temporality
    terms: dict[str, str] = field(default_factory=dict)
    initial_condition: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class WeakFormEquation:
    """A weak-form escape-hatch equation (§1.5).

    `form` is a raw string holding the residual expression (the equation
    reads `form = 0` for all admissible test functions). For
    `temporality = time_dependent`, the form must contain `partial_t(<variable>)`
    (§1.5.4); the validator enforces this.
    """

    template: Literal["weak_form"] = "weak_form"
    variable: str
    subdomain: str
    temporality: Temporality
    form: str
    initial_condition: str | None = None


Equation: TypeAlias = TemplateEquation | WeakFormEquation


# ---------------------------------------------------------------------------
# Boundary condition (§2.2.6) — five kinds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class BCDirichlet:
    """u = g(x, t) on the labelled boundary (§1.6.2)."""

    kind: Literal["dirichlet"] = "dirichlet"
    variable: str
    boundary: str
    expression: str


@dataclass(frozen=True, kw_only=True, slots=True)
class BCNeumann:
    """D ∇u · n = h(x, t) on the labelled boundary; outward normal is the
    variable's home-subdomain outward normal (§1.6.2, §1.6.3)."""

    kind: Literal["neumann"] = "neumann"
    variable: str
    boundary: str
    expression: str


@dataclass(frozen=True, kw_only=True, slots=True)
class BCRobin:
    """α u + β D ∇u · n = h on the labelled boundary (§1.6.2).

    Three separate scalar expressions, per the schema in §2.2.6. (Part 1's
    earlier "tuple" wording was aligned to this representation in the
    review pass.)
    """

    kind: Literal["robin"] = "robin"
    variable: str
    boundary: str
    alpha: str
    beta: str
    expression: str


@dataclass(frozen=True, kw_only=True, slots=True)
class BCInterfaceValueEquality:
    """u_L = k · u_R at an internal boundary (§1.6.2).

    `partner_variable` is the variable on the other side; `expression`
    carries the partition coefficient k (defaults to "1" for pure
    continuity).
    """

    kind: Literal["interface_value_equality"] = "interface_value_equality"
    variable: str
    partner_variable: str
    boundary: str
    expression: str = "1"


@dataclass(frozen=True, kw_only=True, slots=True)
class BCInterfaceFluxBalance:
    """D ∇u_L · n = f(traces, params) at a bulk-bulk internal boundary
    (§1.6.2, post-review-pass clarification).

    The partner side's equal-and-opposite flux is implicit by mass
    conservation; the partner variable is named so `expression` can
    reference it. This kind is bulk-bulk only — bulk-surface accumulation
    uses the composable Neumann + source pattern in §1.6.5, not this kind.
    The validator (§1.11.7) rejects flux-balance entries whose variable or
    partner_variable lives on a non-volume subdomain.
    """

    kind: Literal["interface_flux_balance"] = "interface_flux_balance"
    variable: str
    partner_variable: str
    boundary: str
    expression: str


BoundaryCondition: TypeAlias = BCDirichlet | BCNeumann | BCRobin | BCInterfaceValueEquality | BCInterfaceFluxBalance


# ---------------------------------------------------------------------------
# MathDescription (§2.1.2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class MathDescription:
    """The top-level declarative artifact (§2.1.2).

    `geometry` is a *name* the loader resolves to a concrete Geometry object
    (§3.4 "Name resolution"); it is not a file path. `subdomains`,
    `variables`, and `equations` are required and non-empty; `parameters`
    and `boundary_conditions` default to empty lists per the review pass.
    """

    geometry: str
    subdomains: list[Subdomain]
    variables: list[Variable]
    equations: list[Equation]
    parameters: list[Parameter] = field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
