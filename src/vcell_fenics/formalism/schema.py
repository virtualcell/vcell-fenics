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

The unions also carry pydantic discriminators (`Field(discriminator=...)` or a
callable `Discriminator`) so a `pydantic.TypeAdapter` can validate this tree at
the YAML/JSON boundary (`formalism.loader`). `BoundaryCondition` is a clean
Literal-tagged union; `Equation`, `Motion`, and `Parameter` dispatch on an open
tag or on field presence (the `{name, value}` parameter shorthand omits `kind`),
so they use callable discriminators that also raise the domain errors. The
shared `_CONFIG` rejects unknown fields and coerces numeric YAML scalars to the
string fields (e.g. `initial_condition: 0`), matching the old hand loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypeAlias

from pydantic import BeforeValidator, ConfigDict, Discriminator, Field, Tag

# extra="forbid": reject unknown fields (the old loader's _reject_unknown).
# coerce_numbers_to_str: a YAML int/float in a string field (initial_condition,
# terms values, expressions) becomes its str form, as the old loader's str(...) did.
_CONFIG = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

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


def _validate_motion(value: object) -> object:
    """Enforce the motion domain rules the old loader checked. Raised in a BeforeValidator
    (not the discriminator, whose raises would escape un-wrapped) so the error carries the
    `subdomains[i].motion` location."""

    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "prescribed":
            has_v, has_d = "velocity" in value, "displacement" in value
            if has_v and has_d:
                raise ValueError("prescribed motion must declare exactly one of 'velocity' or 'displacement', not both")
            if not (has_v or has_d):
                raise ValueError("prescribed motion requires either 'velocity' or 'displacement'")
        elif kind not in ("none", "unknown"):
            raise ValueError(f"unknown motion kind {kind!r}; expected 'none', 'prescribed' or 'unknown'")
    return value


def _motion_tag(value: object) -> str:
    """Route a (validated) Motion to its concrete class. `velocity` / `displacement` both
    carry kind='prescribed' and are split by which field is present."""

    if isinstance(value, dict):
        if value.get("kind") == "prescribed":
            return "velocity" if "velocity" in value else "displacement"
        return str(value.get("kind"))
    if isinstance(value, MotionPrescribedVelocity):
        return "velocity"
    if isinstance(value, MotionPrescribedDisplacement):
        return "displacement"
    if isinstance(value, MotionNone | MotionUnknown):
        return value.kind
    raise ValueError(f"cannot determine motion kind for {type(value).__name__}")


_MotionUnion: TypeAlias = Annotated[
    Annotated[MotionNone, Tag("none")]
    | Annotated[MotionPrescribedVelocity, Tag("velocity")]
    | Annotated[MotionPrescribedDisplacement, Tag("displacement")]
    | Annotated[MotionUnknown, Tag("unknown")],
    Discriminator(_motion_tag),
]
Motion: TypeAlias = Annotated[_MotionUnion, BeforeValidator(_validate_motion)]


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
    expression body references any subdomain-relative geometry quantity
    (geom.normal, geom.mean_curvature, geom.azimuth, …); the
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
    values: Annotated[dict[str, float], Field(min_length=1)]


def _validate_parameter(value: object) -> object:
    """Enforce the parameter dispatch rules the old loader checked: `value`/`expression`
    are mutually exclusive, an explicit `kind` must match the field shape, and one of the
    three forms must be present. Raised in a BeforeValidator so the error carries the
    `parameters[i]` location (a discriminator's raise would escape un-wrapped)."""

    if isinstance(value, dict):
        kind = value.get("kind")
        if kind != "region_map":
            has_e, has_val = "expression" in value, "value" in value
            if has_e and has_val:
                raise ValueError(
                    "parameter cannot declare both 'expression' and 'value' (those are mutually exclusive forms)"
                )
            if has_e and kind not in (None, "expression"):
                raise ValueError(
                    f"expression-valued parameter cannot have kind={kind!r}; expected 'expression' or omitted"
                )
            if has_val and kind not in (None, "scalar"):
                raise ValueError(f"constant parameter cannot have kind={kind!r}; expected 'scalar' or omitted")
            if not (has_e or has_val):
                raise ValueError(
                    "parameter must have one of: 'value' (constant), 'expression' (expression-valued), "
                    "or 'kind: region_map' with 'values' (region-keyed)"
                )
    return value


def _parameter_tag(value: object) -> str:
    """Route a (validated) Parameter to its concrete class. The `{name, value}` /
    `{name, expression}` shorthands omit `kind`, so dispatch is by field presence."""

    if isinstance(value, dict):
        if value.get("kind") == "region_map":
            return "region_map"
        return "expression" if "expression" in value else "scalar"
    if isinstance(value, ParameterExpression):
        return "expression"
    if isinstance(value, ParameterRegionMap):
        return "region_map"
    if isinstance(value, ParameterConstant):
        return "scalar"
    raise ValueError(f"cannot determine parameter kind for {type(value).__name__}")


_ParameterUnion: TypeAlias = Annotated[
    Annotated[ParameterConstant, Tag("scalar")]
    | Annotated[ParameterExpression, Tag("expression")]
    | Annotated[ParameterRegionMap, Tag("region_map")],
    Discriminator(_parameter_tag),
]
Parameter: TypeAlias = Annotated[_ParameterUnion, BeforeValidator(_validate_parameter)]


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


def _equation_tag(value: object) -> str:
    """Route an Equation: `template: weak_form` is the WeakFormEquation; any other
    template name is a TemplateEquation (whose `template` is an open registry name)."""

    if isinstance(value, dict):
        return "weak_form" if value.get("template") == "weak_form" else "template"
    return "weak_form" if isinstance(value, WeakFormEquation) else "template"


Equation: TypeAlias = Annotated[
    Annotated[WeakFormEquation, Tag("weak_form")] | Annotated[TemplateEquation, Tag("template")],
    Discriminator(_equation_tag),
]


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
    """u = k · u_adjacent at an internal boundary (§1.6.2).

    `adjacent_variable` is the bulk variable on the other side of the interface
    (the membrane is between the two compartments); `expression` carries the
    partition coefficient k (defaults to "1" for pure continuity).
    """

    kind: Literal["interface_value_equality"] = "interface_value_equality"
    variable: str
    adjacent_variable: str
    boundary: str
    expression: str = "1"


@dataclass(frozen=True, kw_only=True, slots=True)
class BCInterfaceFlux:
    """A **single-sided** Neumann flux on a bulk variable at an internal membrane interface:
    ``D ∇u·n = f`` on `variable`'s side of `boundary`, where `f` is an arbitrary (possibly nonlinear)
    function of the adjacent traces — the variable's own ``trace(u)`` and the ``trace(·)`` of any bulk
    variable in the adjacent compartment across the membrane — plus membrane variables, parameters,
    coordinates and time.

    This is VCell's general "jump condition" side (`in_flux` / `out_flux`), faithfully: the two sides
    of a membrane are **independent** single-sided fluxes, not an enforced equal-and-opposite balance
    (mass conservation is whatever the modeller writes into the two sides). A species crossing the
    membrane is then two ``BCInterfaceFlux`` entries — one per species, each carrying the flux into its
    own side. Distinct from ``BCNeumann`` (an *external* boundary of a single subdomain, which cannot
    reach an adjacent-compartment trace or a membrane variable); the coupled assembler binds those for
    this kind."""

    kind: Literal["interface_flux"] = "interface_flux"
    variable: str
    boundary: str
    expression: str


# Clean Literal-tagged union — every member has a unique `kind`, so pydantic
# discriminates on the field directly (no callable needed).
BoundaryCondition: TypeAlias = Annotated[
    BCDirichlet | BCNeumann | BCRobin | BCInterfaceValueEquality | BCInterfaceFlux,
    Field(discriminator="kind"),
]


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
    subdomains: Annotated[list[Subdomain], Field(min_length=1)]
    variables: Annotated[list[Variable], Field(min_length=1)]
    equations: Annotated[list[Equation], Field(min_length=1)]
    parameters: list[Parameter] = field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)


# Attach the shared pydantic config to every dataclass so a `pydantic.TypeAdapter`
# (the loader) rejects unknown fields and coerces numeric scalars into string fields.
# Config can't be passed to TypeAdapter for a dataclass — it must live on the class —
# and it does not propagate to nested types, so each class carries it. The classes stay
# plain stdlib dataclasses; this attribute is inert unless pydantic validates them.
_CONFIGURED_CLASSES: tuple[type, ...] = (
    MotionNone,
    MotionPrescribedVelocity,
    MotionPrescribedDisplacement,
    MotionUnknown,
    Subdomain,
    Variable,
    ParameterConstant,
    ParameterExpression,
    ParameterRegionMap,
    TemplateEquation,
    WeakFormEquation,
    BCDirichlet,
    BCNeumann,
    BCRobin,
    BCInterfaceValueEquality,
    BCInterfaceFlux,
    MathDescription,
)
for _cls in _CONFIGURED_CLASSES:
    # pydantic reads `__pydantic_config__` off the class to validate these stdlib dataclasses;
    # it is not a known attribute of `type`, hence the targeted ignore.
    _cls.__pydantic_config__ = _CONFIG  # type: ignore[attr-defined]
