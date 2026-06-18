"""Validation pass over a MathDescription (docs/modeling/declarative-formalism.md §2.5).

This module implements the checks of §1.11 that do not need a Geometry object.
Schema-level checks read declared fields; expression-level checks parse each
expression string (via `formalism.parser`) and walk the AST. Two things stay
deferred: the geometry cross-check (§1.11.10, which needs a Geometry the
MathDescription does not carry) and BC-expression contents (a BC evaluates on a
boundary whose incident subdomains are geometry-side, so its resolution context
is not known intra-model).

Schema-level checks:

- Reserved-name shadowing and duplicate / colliding declarations (§1.11.3).
- Structural reference resolution — subdomains, governed variables, motion
  variables, BC variables (§1.11.3).
- Coverage — every variable governed by exactly one equation; the IC-presence
  ⇔ temporality rule, with the unknown-motion-variable exception (§1.11.4).
- Template conformance — known template, allowed temporality / subdomain kind /
  governed type, legal and required slots, T1's "at least one of
  diffusion / source" (§1.4.2).
- Boundary-condition consistency — conflicting kinds, interface-partner
  declaration, `interface_flux_balance` bulk-only, weak-form Dirichlet-only
  (§1.11.7).

Expression-level checks (parse + AST walk):

- Name resolution — every bare name resolves to a local variable, parameter,
  `t`/`x`, or (in weak forms) a measure or `<var>_test`; cross-subdomain refs
  require `trace`, and `trace` only crosses high → low dimension (§1.8.2,
  §1.8.6, §1.11.3).
- Parameter expressions — no variable references, acyclic parameter graph, a
  `subdomain:` scope when geometric helpers appear, and scope-compatible use
  sites (§1.8.3, §1.11.3, §2.2.3).
- Initial conditions — no references to `sim.t` or to state variables (§1.11.8).
- Weak forms — `partial_t(governed)` present iff `time_dependent`, and the
  `<governed>_test` function referenced at least once (§1.9.5, §2.3.5).
- Operator usage — the narrow rule (no calculus on the slot's own variable) and
  the smoothness rule (`lapl` / `lapl_beltrami` need a P2+ argument) (§1.11.9).
- Type-checking (§1.11.5) — bottom-up type inference over the vocabulary; each
  term-slot, initial-condition, motion, and expression-parameter expression
  must produce the type its slot declares, with no implicit broadcast (§1.8.8).
  The numeric literal `0` is the one exception: it is the zero of any expected
  type. Weak-form `form:` residuals are not type-checked (escape hatch, §1.5).

Diagnostics are collected (not fail-fast) so a single pass reports every
structural problem in a large model. `validate` returns all of them;
`validate_or_raise` raises `FormalismValidationError` if any is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, assert_never

from vcell_fenics.formalism.expr import (
    BinaryOp,
    Expr,
    FunctionCall,
    IndexAccess,
    Name,
    Number,
    TensorLiteral,
    UnaryOp,
    VectorLiteral,
)
from vcell_fenics.formalism.parser import ExpressionSyntaxError, parse
from vcell_fenics.formalism.schema import (
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    BoundaryCondition,
    Equation,
    MathDescription,
    MotionPrescribedDisplacement,
    MotionPrescribedVelocity,
    MotionUnknown,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    ParameterRegionMap,
    Subdomain,
    TemplateEquation,
    Variable,
    VariableType,
    WeakFormEquation,
)
from vcell_fenics.formalism.templates import REGISTRY
from vcell_fenics.formalism.vocabulary import (
    CALCULUS_OPERATORS,
    GEOMETRY_MEMBERS,
    MEASURES,
    QUALIFIED_BUILTINS,
    RESERVED_CALLABLES,
    RESERVED_NAMES,
    SCOPED_GEOMETRY_NAMES,
    SIMULATION_MEMBERS,
    STANDARD_FUNCTIONS,
    TENSOR_ALGEBRA,
)

Severity = Literal["error", "warning"]

# Relative topological dimension of each subdomain kind, volume highest. Lets
# the trace direction rule (§1.8.2) compare dimensions without the ambient d.
_KIND_RANK: dict[str, int] = {"point": 0, "curve": 1, "surface": 2, "volume": 3}

# Names that may appear only as a call's callee, never as a bare value (operators/functions).
_CALLABLE_NAMES: frozenset[str] = RESERVED_CALLABLES

_ExprKind = Literal["motion", "parameter", "term_slot", "initial_condition", "weak_form"]

# Inferred type of an expression node. "error" is a sentinel for an
# unresolvable / already-diagnosed subtree; it suppresses cascade errors.
_Type = Literal["scalar", "vector", "symmetric_tensor", "error"]

_VECTOR_ONLY: frozenset[VariableType] = frozenset({"vector"})

# (argument type, result type) for each calculus operator (§1.8.5).
_CALCULUS_SIGNATURE: dict[str, tuple[_Type, _Type]] = {
    "grad": ("scalar", "vector"),
    "div": ("vector", "scalar"),
    "lapl": ("scalar", "scalar"),
    "grad_surf": ("scalar", "vector"),
    "div_surf": ("vector", "scalar"),
    "lapl_beltrami": ("scalar", "scalar"),
}


def _space_admits_second_derivative(space: str) -> bool:
    """Whether a function-space hint admits a meaningful strong second
    derivative (§1.11.9). `lagrange_pN` needs N ≥ 2; unknown hints are given the
    benefit of the doubt since the rule targets the P1 default specifically."""

    match = re.fullmatch(r"lagrange_p(\d+)", space)
    if match is not None:
        return int(match.group(1)) >= 2
    return True


def _references_divergence(node: Expr) -> bool:
    """Whether the expression contains a divergence operator (`div` / `div_surf`) anywhere — the
    structural marker of a dilution term `ρ ∇_Γ·v_Γ`."""

    if isinstance(node, FunctionCall):
        return node.callee in ("div", "div_surf") or any(_references_divergence(a) for a in node.args)
    if isinstance(node, BinaryOp):
        return _references_divergence(node.left) or _references_divergence(node.right)
    if isinstance(node, UnaryOp):
        return _references_divergence(node.operand)
    if isinstance(node, IndexAccess):
        return _references_divergence(node.base) or _references_divergence(node.index)
    if isinstance(node, VectorLiteral):
        return any(_references_divergence(c) for c in node.components)
    if isinstance(node, TensorLiteral):
        return any(_references_divergence(row) for row in node.rows)
    return False  # Name, Number — no operator


def _div_of_vector_variable(node: Expr, vector_vars: frozenset[str]) -> str | None:
    """The name of a vector variable appearing as a bare `div(v)` / `div_surf(v)` — the
    incompressibility-constraint signature `∫ q ∇·v` of a saddle point — or None. Requires the
    argument to be the bare variable (not an expression like `div(c*v)` or `div(-D grad(c))`, so
    a conservative-flux transport term is not mistaken for a pressure constraint)."""

    if isinstance(node, FunctionCall):
        if (
            node.callee in ("div", "div_surf")
            and len(node.args) == 1
            and isinstance(node.args[0], Name)
            and node.args[0].name in vector_vars
        ):
            return node.args[0].name
        for argument in node.args:
            found = _div_of_vector_variable(argument, vector_vars)
            if found is not None:
                return found
        return None
    if isinstance(node, BinaryOp):
        return _div_of_vector_variable(node.left, vector_vars) or _div_of_vector_variable(node.right, vector_vars)
    if isinstance(node, UnaryOp):
        return _div_of_vector_variable(node.operand, vector_vars)
    if isinstance(node, IndexAccess):
        return _div_of_vector_variable(node.base, vector_vars) or _div_of_vector_variable(node.index, vector_vars)
    if isinstance(node, VectorLiteral):
        return next((r for c in node.components if (r := _div_of_vector_variable(c, vector_vars))), None)
    if isinstance(node, TensorLiteral):
        return next((r for row in node.rows if (r := _div_of_vector_variable(row, vector_vars))), None)
    return None


def _references_partial_t_of(node: Expr, variable: str) -> bool:
    """Whether the expression contains `partial_t(variable)` — i.e. the variable evolves in time
    (so it is not a pure Lagrange-multiplier constraint like a pressure)."""

    if isinstance(node, FunctionCall):
        if node.callee == "partial_t" and len(node.args) == 1 and isinstance(node.args[0], Name):
            return node.args[0].name == variable
        return any(_references_partial_t_of(a, variable) for a in node.args)
    if isinstance(node, BinaryOp):
        return _references_partial_t_of(node.left, variable) or _references_partial_t_of(node.right, variable)
    if isinstance(node, UnaryOp):
        return _references_partial_t_of(node.operand, variable)
    if isinstance(node, IndexAccess):
        return _references_partial_t_of(node.base, variable) or _references_partial_t_of(node.index, variable)
    if isinstance(node, VectorLiteral):
        return any(_references_partial_t_of(c, variable) for c in node.components)
    if isinstance(node, TensorLiteral):
        return any(_references_partial_t_of(row, variable) for row in node.rows)
    return False


def _lagrange_order(space: str) -> int | None:
    """The polynomial order of a `lagrange_pN` space hint, or None for an unrecognised hint."""

    match = re.fullmatch(r"lagrange_p(\d+)", space)
    return int(match.group(1)) if match is not None else None


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One validation finding. `path` is an attribute path into the
    MathDescription (e.g. `equations[1].terms`); empty for whole-model
    findings."""

    severity: Severity
    path: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}: " if self.path else ""
        return f"[{self.severity}] {where}{self.message}"


class FormalismValidationError(Exception):
    """Raised by `validate_or_raise` when a MathDescription has validation
    errors. `errors` holds the error-severity diagnostics; the str() lists
    them one per line."""

    def __init__(self, errors: list[Diagnostic]) -> None:
        self.errors = errors
        body = "\n".join(f"  {d}" for d in errors)
        super().__init__(f"MathDescription failed validation with {len(errors)} error(s):\n{body}")


@dataclass(frozen=True, slots=True)
class _ExprContext:
    """The evaluation context of one expression string, threaded through the
    AST walk so each node knows where it lives and what it may reference."""

    path: str
    eval_subdomain: str | None
    kind: _ExprKind
    governed_var: str | None


def validate(md: MathDescription) -> list[Diagnostic]:
    """Run the structural validation pass and return all diagnostics
    (errors and warnings), in document order."""

    return _Validator(md).run()


def validate_or_raise(md: MathDescription) -> list[Diagnostic]:
    """Validate and raise `FormalismValidationError` if any error is found.
    Returns the (possibly empty) list of warning diagnostics on success."""

    diagnostics = validate(md)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        raise FormalismValidationError(errors)
    return [d for d in diagnostics if d.severity == "warning"]


class _Validator:
    def __init__(self, md: MathDescription) -> None:
        self._md = md
        self._diagnostics: list[Diagnostic] = []

        # Indices, built with duplicate detection.
        self._subdomain_by_name: dict[str, Subdomain] = {}
        self._var_by_key: dict[tuple[str, str], Variable] = {}
        self._var_subdomains: dict[str, set[str]] = {}
        self._param_names: set[str] = set()
        self._param_by_name: dict[str, Parameter] = {}
        # (variable, subdomain) pairs whose subdomain has unknown motion driven
        # by that variable — the IC-on-steady-state exception (§1.7.7).
        self._unknown_motion_vars: set[tuple[str, str]] = set()

    # -- diagnostics ---------------------------------------------------------

    def _error(self, path: str, message: str) -> None:
        self._diagnostics.append(Diagnostic("error", path, message))

    def _warn(self, path: str, message: str) -> None:
        self._diagnostics.append(Diagnostic("warning", path, message))

    # -- driver --------------------------------------------------------------

    def run(self) -> list[Diagnostic]:
        self._build_indices()
        self._check_reserved_and_collisions()
        self._check_parameter_subdomain_refs()
        self._check_motion_variables()
        self._check_equations()
        self._check_boundary_conditions()
        self._check_expressions()
        self._check_dilution_on_moving_weak_forms()
        self._check_inf_sup_element_pair()
        return self._diagnostics

    def _build_indices(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if s.name in self._subdomain_by_name:
                self._error(f"subdomains[{i}]", f"duplicate subdomain name {s.name!r}")
            else:
                self._subdomain_by_name[s.name] = s

        for i, v in enumerate(self._md.variables):
            key = (v.name, v.subdomain)
            if key in self._var_by_key:
                self._error(f"variables[{i}]", f"duplicate variable {v.name!r} on subdomain {v.subdomain!r}")
            else:
                self._var_by_key[key] = v
            self._var_subdomains.setdefault(v.name, set()).add(v.subdomain)

        for i, p in enumerate(self._md.parameters):
            if p.name in self._param_names:
                self._error(f"parameters[{i}]", f"duplicate parameter name {p.name!r}")
            else:
                self._param_names.add(p.name)
                self._param_by_name[p.name] = p

        for s in self._md.subdomains:
            if isinstance(s.motion, MotionUnknown):
                self._unknown_motion_vars.add((s.motion.variable, s.name))

    # -- §1.11.3 reserved names, shadowing, collisions -----------------------

    def _check_reserved_and_collisions(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if s.name in RESERVED_NAMES:
                self._error(f"subdomains[{i}]", f"subdomain name {s.name!r} is a reserved name (§2.4.1)")
        for i, v in enumerate(self._md.variables):
            if v.name in RESERVED_NAMES:
                self._error(f"variables[{i}]", f"variable name {v.name!r} is a reserved name (§2.4.1)")
            if v.subdomain not in self._subdomain_by_name:
                self._error(f"variables[{i}]", f"references undeclared subdomain {v.subdomain!r}")
        for i, p in enumerate(self._md.parameters):
            if p.name in RESERVED_NAMES:
                self._error(f"parameters[{i}]", f"parameter name {p.name!r} is a reserved name (§2.4.1)")
            elif p.name in self._var_subdomains:
                self._error(
                    f"parameters[{i}]",
                    f"parameter {p.name!r} shadows a variable of the same name; shadowing is an error (§1.8.6)",
                )

    def _check_parameter_subdomain_refs(self) -> None:
        for i, p in enumerate(self._md.parameters):
            scope: str | None = None
            if isinstance(p, ParameterExpression | ParameterRegionMap):
                scope = p.subdomain
            if scope is not None and scope not in self._subdomain_by_name:
                self._error(f"parameters[{i}]", f"references undeclared subdomain {scope!r}")

    def _check_motion_variables(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if not isinstance(s.motion, MotionUnknown):
                continue
            key = (s.motion.variable, s.name)
            var = self._var_by_key.get(key)
            if var is None:
                self._error(
                    f"subdomains[{i}].motion",
                    f"motion variable {s.motion.variable!r} is not a declared variable on subdomain {s.name!r}",
                )
            elif var.type != "vector":
                self._error(
                    f"subdomains[{i}].motion",
                    f"motion variable {s.motion.variable!r} must be vector-typed (§1.10.3), got {var.type!r}",
                )

    # -- §1.11.4 coverage + §1.4.2 template conformance ----------------------

    def _check_equations(self) -> None:
        governed_count: dict[tuple[str, str], int] = {}

        for i, eq in enumerate(self._md.equations):
            path = f"equations[{i}]"
            subdomain = self._subdomain_by_name.get(eq.subdomain)
            if subdomain is None:
                self._error(path, f"references undeclared subdomain {eq.subdomain!r}")

            key = (eq.variable, eq.subdomain)
            var = self._var_by_key.get(key)
            if var is None:
                self._error(path, f"governs variable {eq.variable!r}, not declared on subdomain {eq.subdomain!r}")
            governed_count[key] = governed_count.get(key, 0) + 1

            self._check_ic_temporality(eq, key, path)

            if not isinstance(eq, WeakFormEquation):
                self._check_template_equation(eq, var, subdomain, path)

        for v in self._md.variables:
            count = governed_count.get((v.name, v.subdomain), 0)
            where = f"variable {v.name!r} on subdomain {v.subdomain!r}"
            if count == 0:
                self._error("equations", f"{where} is not governed by any equation (undetermined)")
            elif count > 1:
                self._error("equations", f"{where} is governed by {count} equations (overdetermined)")

    def _check_ic_temporality(self, eq: Equation, key: tuple[str, str], path: str) -> None:
        if eq.temporality == "time_dependent":
            if eq.initial_condition is None:
                self._error(path, "time_dependent equation requires an initial_condition (§1.9.5)")
        elif eq.initial_condition is not None and key not in self._unknown_motion_vars:
            self._error(
                path,
                "steady_state equation must not have an initial_condition (§1.9.5); the only exception is an "
                "unknown-motion variable, whose IC sets the t=0 configuration (§1.7.7)",
            )

    def _check_template_equation(
        self, eq: TemplateEquation, var: Variable | None, subdomain: Subdomain | None, path: str
    ) -> None:
        template = eq.template
        terms = eq.terms
        temporality = eq.temporality
        spec = REGISTRY.get(template)
        if spec is None:
            options = sorted(REGISTRY)
            self._error(path, f"unknown template {template!r} (v1 templates: {options}, or 'weak_form')")
            return

        if temporality not in spec.temporalities:
            self._error(
                path,
                f"template {template!r} does not support temporality {temporality!r} "
                f"(allowed: {sorted(spec.temporalities)})",
            )
        if subdomain is not None and subdomain.kind not in spec.subdomain_kinds:
            self._error(
                path,
                f"template {template!r} requires subdomain kind in {sorted(spec.subdomain_kinds)}; "
                f"subdomain {subdomain.name!r} is {subdomain.kind!r}",
            )
        if var is not None and var.type not in spec.governed_types:
            self._error(
                path,
                f"template {template!r} governs a {sorted(spec.governed_types)} variable; {var.name!r} is {var.type!r}",
            )

        valid_slots = {s.name for s in spec.slots}
        for slot_name in terms:
            if slot_name not in valid_slots:
                self._error(
                    f"{path}.terms",
                    f"unknown slot {slot_name!r} for template {template!r} (slots: {sorted(valid_slots)})",
                )
        for slot in spec.slots:
            if slot.required and slot.name not in terms:
                self._error(f"{path}.terms", f"template {template!r} requires slot {slot.name!r}")
        if spec.require_any_of and not any(name in terms for name in spec.require_any_of):
            self._error(
                f"{path}.terms",
                f"template {template!r} requires at least one of {list(spec.require_any_of)}",
            )

    # -- §1.11.7 boundary-condition consistency ------------------------------

    def _check_dilution_on_moving_weak_forms(self) -> None:
        """Registry #1 of docs/modeling/validation-and-diagnostics.md: a co-moving density on a
        moving subdomain needs the dilution term `ρ ∇_Γ·v_Γ` (bulk: `c ∇·v`) for mass
        conservation. A *templated* equation gets it automatically (the backend adds it whenever
        the subdomain moves); a *weak-form* escape-hatch equation does not. So warn when a
        time-dependent, scalar weak form on a moving subdomain references no divergence operator
        at all — a strong sign the dilution term was forgotten. Heuristic (the term could in
        principle be written without an explicit `div`), so a **warning**, not an error; and
        gated to scalar densities, since a vector weak form is a momentum balance that needs no
        dilution."""

        moving = (MotionPrescribedVelocity, MotionPrescribedDisplacement, MotionUnknown)
        for i, eq in enumerate(self._md.equations):
            if not isinstance(eq, WeakFormEquation) or eq.temporality != "time_dependent":
                continue
            subdomain = self._subdomain_by_name.get(eq.subdomain)
            if subdomain is None or not isinstance(subdomain.motion, moving):
                continue
            variable = self._var_by_key.get((eq.variable, eq.subdomain))
            if variable is None or variable.type != "scalar":  # dilution is for densities, not a momentum balance
                continue
            form = self._parse(eq.form, f"equations[{i}].form")
            if form is None or _references_divergence(form):
                continue
            self._warn(
                f"equations[{i}].form",
                f"weak form for the density {eq.variable!r} on the moving subdomain "
                f"{eq.subdomain!r} references no divergence operator — a co-moving density needs a "
                f"dilution term (e.g. {eq.variable} * div_surf(<motion velocity>) * "
                f"{eq.variable}_test, inlining the subdomain's velocity) for mass conservation, "
                f"which the backend does not add to a weak form (unlike a template). Confirm it is "
                f"intended.",
            )

    def _check_inf_sup_element_pair(self) -> None:
        """Registry #6 of docs/modeling/validation-and-diagnostics.md: an incompressible saddle
        point (a pressure enforcing `∇·v = 0`) needs an **inf-sup-stable** velocity/pressure pair
        — the velocity a higher-order space than the pressure (Taylor–Hood, e.g. P2/P1). An
        equal-order pair gives spurious pressure oscillations unless stabilised.

        The saddle-point pressure is detected structurally: a *scalar* weak form whose residual
        contains a bare `div(v)`/`div_surf(v)` for a *vector variable* `v` and does **not** evolve
        the pressure in time (no `partial_t(p)` — it is a constraint, not a transported density).
        A warning (the inference is structural); both elements are read from the `space` hints."""

        vector_vars_by_subdomain: dict[str, frozenset[str]] = {}
        for (name, subdomain), var in self._var_by_key.items():
            if var.type == "vector":
                vector_vars_by_subdomain[subdomain] = vector_vars_by_subdomain.get(subdomain, frozenset()) | {name}

        for i, eq in enumerate(self._md.equations):
            if not isinstance(eq, WeakFormEquation):
                continue
            pressure = self._var_by_key.get((eq.variable, eq.subdomain))
            if pressure is None or pressure.type != "scalar":
                continue
            form = self._parse(eq.form, f"equations[{i}].form")
            if form is None or _references_partial_t_of(form, eq.variable):
                continue  # an evolving scalar is a density, not a pressure multiplier
            velocity_name = _div_of_vector_variable(form, vector_vars_by_subdomain.get(eq.subdomain, frozenset()))
            if velocity_name is None:
                continue
            velocity = self._var_by_key[(velocity_name, eq.subdomain)]
            velocity_order, pressure_order = _lagrange_order(velocity.space), _lagrange_order(pressure.space)
            if velocity_order is None or pressure_order is None or velocity_order > pressure_order:
                continue  # unrecognised hint, or a stable Taylor–Hood pair
            self._warn(
                f"equations[{i}].form",
                f"the velocity {velocity_name!r} ({velocity.space}) and pressure {eq.variable!r} "
                f"({pressure.space}) form an incompressible saddle point but are not an "
                f"inf-sup-stable pair — the velocity must be a higher-order space than the "
                f"pressure (Taylor–Hood, e.g. lagrange_p2 velocity / lagrange_p1 pressure). An "
                f"equal-order pair gives spurious pressure oscillations unless stabilised.",
            )

    def _check_boundary_conditions(self) -> None:
        weak_form_vars = {(eq.variable, eq.subdomain) for eq in self._md.equations if isinstance(eq, WeakFormEquation)}
        groups: dict[tuple[str, str], set[str]] = {}

        for i, bc in enumerate(self._md.boundary_conditions):
            path = f"boundary_conditions[{i}]"
            if bc.variable not in self._var_subdomains:
                self._error(path, f"references undeclared variable {bc.variable!r}")
            if (
                isinstance(bc, BCInterfaceValueEquality | BCInterfaceFluxBalance)
                and bc.partner_variable not in self._var_subdomains
            ):
                self._error(path, f"partner_variable {bc.partner_variable!r} is not declared")
            if isinstance(bc, BCInterfaceFluxBalance):
                self._check_flux_balance_bulk_only(bc, path)
            self._check_weak_form_dirichlet_only(bc, path, weak_form_vars)
            groups.setdefault((bc.variable, bc.boundary), set()).add(bc.kind)

        for (variable, boundary), kinds in groups.items():
            if len(kinds) > 1:
                self._error(
                    "boundary_conditions",
                    f"conflicting BC kinds {sorted(kinds)} for variable {variable!r} on boundary {boundary!r}",
                )

    def _check_flux_balance_bulk_only(self, bc: BCInterfaceFluxBalance, path: str) -> None:
        for role, name in (("variable", bc.variable), ("partner_variable", bc.partner_variable)):
            hosts = self._var_subdomains.get(name)
            if not hosts:
                continue
            kinds = {self._subdomain_by_name[s].kind for s in hosts if s in self._subdomain_by_name}
            if kinds and "volume" not in kinds:
                self._error(
                    path,
                    f"interface_flux_balance requires {role} {name!r} to live on a volume subdomain; it lives on "
                    f"{sorted(kinds)}. Bulk-surface coupling uses the §1.6.5 Neumann + source pattern instead",
                )

    def _check_weak_form_dirichlet_only(
        self, bc: BoundaryCondition, path: str, weak_form_vars: set[tuple[str, str]]
    ) -> None:
        variable = bc.variable
        hosts = self._var_subdomains.get(variable)
        if not hosts or bc.kind == "dirichlet":
            return
        # Only flag when the variable is weak-form-governed on *every* subdomain
        # it lives on; otherwise the offending subdomain is ambiguous without a
        # Geometry, and a template-governed home could legitimately take this BC.
        if all((variable, s) in weak_form_vars for s in hosts):
            self._error(
                path,
                f"variable {variable!r} is governed by a weak-form equation; only Dirichlet BCs are allowed on it "
                f"(§1.5.6) — encode natural BCs in the form itself",
            )

    # -- §1.11.3/§1.11.8/§1.11.9 expression-level checks ---------------------

    def _check_expressions(self) -> None:
        param_trees: dict[str, Expr] = {}
        for i, p in enumerate(self._md.parameters):
            if not isinstance(p, ParameterExpression):
                continue
            path = f"parameters[{i}].expression"
            tree = self._parse(p.expression, path)
            if tree is None:
                continue
            param_trees[p.name] = tree
            ctx = _ExprContext(path, p.subdomain, "parameter", None)
            self._walk(tree, ctx)
            if p.subdomain is None and self._uses_geometric_helper(tree):
                self._error(
                    path,
                    "expression parameter references a geometric helper but declares no 'subdomain:' scope (§2.2.3)",
                )
            self._check_expr_type(tree, ctx, frozenset({p.type}))
        self._check_parameter_cycles(param_trees)

        for i, s in enumerate(self._md.subdomains):
            motion = s.motion
            if isinstance(motion, MotionPrescribedVelocity):
                self._walk_site(
                    motion.velocity, f"subdomains[{i}].motion.velocity", s.name, "motion", None, _VECTOR_ONLY
                )
            elif isinstance(motion, MotionPrescribedDisplacement):
                self._walk_site(
                    motion.displacement, f"subdomains[{i}].motion.displacement", s.name, "motion", None, _VECTOR_ONLY
                )

        for i, eq in enumerate(self._md.equations):
            path = f"equations[{i}]"
            if isinstance(eq, WeakFormEquation):
                tree = self._parse(eq.form, f"{path}.form")
                if tree is not None:
                    self._walk(tree, _ExprContext(f"{path}.form", eq.subdomain, "weak_form", eq.variable))
                    self._check_weak_form_structure(eq, tree, path)
            else:
                spec = REGISTRY.get(eq.template)
                for slot, text in eq.terms.items():
                    slot_spec = spec.slot(slot) if spec is not None else None
                    expected = slot_spec.types if slot_spec is not None else None
                    self._walk_site(text, f"{path}.terms.{slot}", eq.subdomain, "term_slot", eq.variable, expected)
            if eq.initial_condition is not None:
                var = self._var_by_key.get((eq.variable, eq.subdomain))
                expected = frozenset({var.type}) if var is not None else None
                self._walk_site(
                    eq.initial_condition,
                    f"{path}.initial_condition",
                    eq.subdomain,
                    "initial_condition",
                    eq.variable,
                    expected,
                )

    def _walk_site(
        self,
        text: str,
        path: str,
        subdomain: str | None,
        kind: _ExprKind,
        governed: str | None,
        expected: frozenset[VariableType] | None = None,
    ) -> None:
        tree = self._parse(text, path)
        if tree is None:
            return
        ctx = _ExprContext(path, subdomain, kind, governed)
        self._walk(tree, ctx)
        if expected is not None:
            self._check_expr_type(tree, ctx, expected)

    def _parse(self, text: str, path: str) -> Expr | None:
        try:
            return parse(text)
        except ExpressionSyntaxError as exc:
            self._error(path, f"expression syntax error: {exc.message} (position {exc.pos})")
            return None

    # -- AST walk + per-node resolution / rules ------------------------------

    def _walk(self, node: Expr, ctx: _ExprContext) -> None:
        if isinstance(node, Number):
            return
        if isinstance(node, Name):
            self._resolve_name(node.name, ctx)
            return
        if isinstance(node, IndexAccess):
            self._walk(node.base, ctx)
            self._walk(node.index, ctx)
            return
        if isinstance(node, UnaryOp):
            self._walk(node.operand, ctx)
            return
        if isinstance(node, BinaryOp):
            self._walk(node.left, ctx)
            self._walk(node.right, ctx)
            return
        if isinstance(node, VectorLiteral):
            for component in node.components:
                self._walk(component, ctx)
            return
        if isinstance(node, TensorLiteral):
            for row in node.rows:
                self._walk(row, ctx)
            return
        if isinstance(node, FunctionCall):
            self._walk_call(node, ctx)
            return
        assert_never(node)

    def _walk_call(self, node: FunctionCall, ctx: _ExprContext) -> None:
        callee = node.callee
        weak = ctx.kind == "weak_form"
        if callee == "trace":
            self._check_trace(node, ctx)
            return
        if callee in MEASURES:
            if not weak:
                self._error(ctx.path, f"measure {callee!r} is only valid in a weak-form 'form:' expression (§2.3.4)")
            return  # a measure's argument is a boundary label, not a value expression
        if callee == "partial_t":
            if not weak:
                self._error(ctx.path, "partial_t(...) is only valid in a weak-form 'form:' expression (§2.3.4)")
            for arg in node.args:
                self._walk(arg, ctx)
            return
        if callee in CALCULUS_OPERATORS:
            if (
                ctx.kind == "term_slot"
                and ctx.governed_var is not None
                and any(self._contains_name(arg, ctx.governed_var) for arg in node.args)
            ):
                self._error(
                    ctx.path,
                    f"{callee}(...) may not be applied to the equation's own variable {ctx.governed_var!r} in a "
                    f"template slot (§1.11.9, narrow rule); calculus on other variables is allowed",
                )
            if callee in ("lapl", "lapl_beltrami") and ctx.kind == "term_slot":
                self._check_smoothness(node, ctx)
            for arg in node.args:
                self._walk(arg, ctx)
            return
        if callee in STANDARD_FUNCTIONS or callee in TENSOR_ALGEBRA:
            for arg in node.args:
                self._walk(arg, ctx)
            return
        self._error(ctx.path, f"unknown function {callee!r}")
        for arg in node.args:
            self._walk(arg, ctx)

    def _resolve_name(self, nm: str, ctx: _ExprContext) -> None:
        s = ctx.eval_subdomain
        if ctx.kind == "weak_form":
            if nm in MEASURES:
                return
            if nm.endswith("_test"):
                base = nm[: -len("_test")]
                if s is not None and (base, s) in self._var_by_key:
                    return
                self._error(ctx.path, f"{nm!r} is not a valid test function: no variable {base!r} on subdomain {s!r}")
                return
        if s is not None and (nm, s) in self._var_by_key:
            self._reject_variable_in_restricted_context(nm, ctx)
            return
        if nm in self._param_names:
            self._check_param_use_site(nm, ctx)
            return
        if "." in nm:  # a namespaced built-in: geom.* / sim.* (ADR 006)
            self._resolve_qualified_name(nm, ctx)
            return
        if nm in MEASURES:
            self._error(ctx.path, f"measure {nm!r} is only valid in a weak-form 'form:' expression (§2.3.4)")
            return
        if nm in _CALLABLE_NAMES:
            self._error(ctx.path, f"{nm!r} is a function and must be called with arguments, e.g. {nm}(...)")
            return
        hosts = self._var_subdomains.get(nm)
        if hosts:
            if self._reject_variable_in_restricted_context(nm, ctx):
                return
            higher = sorted(h for h in hosts if self._rank(h) > self._rank(s))
            if higher:
                self._error(
                    ctx.path,
                    f"variable {nm!r} lives on a higher-dimensional subdomain ({higher}); "
                    f"reference it via trace({nm}) (§1.8.2)",
                )
            else:
                self._error(
                    ctx.path,
                    f"variable {nm!r} is defined on a different subdomain ({sorted(hosts)}) and cannot be "
                    f"referenced by bare name here (§1.8.6)",
                )
            return
        self._error(ctx.path, f"unknown name {nm!r}")

    def _resolve_qualified_name(self, nm: str, ctx: _ExprContext) -> None:
        """Validate a namespaced built-in `geom.*` / `sim.*` (ADR 006): reject an unknown member
        and `sim.t` in an initial condition (§1.11.8); otherwise accept. Availability of the
        mechanics-only quantities (`geom.normal` / `geom.mean_curvature`) and the parameter-scope
        rule for subdomain-relative geometry (§2.2.3) are enforced elsewhere."""

        root, _, member = nm.partition(".")
        if root == "geom":
            if member not in GEOMETRY_MEMBERS:
                members = ", ".join(f"geom.{m}" for m in sorted(GEOMETRY_MEMBERS))
                self._error(ctx.path, f"unknown geometry quantity {nm!r}; valid members are {members}")
            return
        if root == "sim":
            if member not in SIMULATION_MEMBERS:
                members = ", ".join(f"sim.{m}" for m in sorted(SIMULATION_MEMBERS))
                self._error(ctx.path, f"unknown simulation quantity {nm!r}; valid members are {members}")
                return
            if member == "t" and ctx.kind == "initial_condition":
                self._error(ctx.path, "initial condition may not reference time sim.t (§1.11.8)")
            return
        self._error(ctx.path, f"unknown name {nm!r}")

    def _reject_variable_in_restricted_context(self, nm: str, ctx: _ExprContext) -> bool:
        """Emit the right diagnostic when a variable is referenced where
        variables are disallowed (parameter expressions, ICs). Returns True iff
        a diagnostic was emitted."""

        if ctx.kind == "parameter":
            self._error(
                ctx.path,
                f"parameter expression may not reference the variable {nm!r}; parameters depend only on sim.t, "
                f"geom.* quantities, and other parameters (§1.8.3)",
            )
            return True
        if ctx.kind == "initial_condition":
            self._error(ctx.path, f"initial condition may not reference the state variable {nm!r} (§1.11.8)")
            return True
        return False

    def _check_trace(self, node: FunctionCall, ctx: _ExprContext) -> None:
        if ctx.kind == "parameter":
            self._error(ctx.path, "parameter expression may not reference variables via trace (§1.8.3)")
            return
        if ctx.kind == "initial_condition":
            self._error(ctx.path, "initial condition may not reference state variables via trace (§1.11.8)")
            return
        if len(node.args) != 1:
            self._error(ctx.path, "trace(...) takes exactly one argument; v1 has no side specifier (§1.8.2)")
            for arg in node.args:
                self._walk(arg, ctx)
            return
        arg = node.args[0]
        if not isinstance(arg, Name):
            self._error(ctx.path, "trace(...) argument must be a variable name (§1.8.2)")
            self._walk(arg, ctx)
            return
        nm = arg.name
        hosts = self._var_subdomains.get(nm, set())
        higher = sorted(h for h in hosts if self._rank(h) > self._rank(ctx.eval_subdomain))
        if not hosts:
            self._error(ctx.path, f"trace argument {nm!r} is not a declared variable")
        elif not higher:
            self._error(
                ctx.path,
                f"trace({nm}) is invalid: {nm!r} does not live on a subdomain of higher dimension than "
                f"{ctx.eval_subdomain!r}; trace only crosses from higher to lower dimension (§1.8.2)",
            )

    def _check_smoothness(self, node: FunctionCall, ctx: _ExprContext) -> None:
        names: set[str] = set()
        for arg in node.args:
            names |= self._local_variable_names(arg, ctx.eval_subdomain)
        for nm in sorted(names):
            var = self._var_by_key.get((nm, ctx.eval_subdomain)) if ctx.eval_subdomain is not None else None
            if var is not None and not _space_admits_second_derivative(var.space):
                self._error(
                    ctx.path,
                    f"{node.callee}({nm}) requires {nm!r} to use 'lagrange_p2' or higher; its space is "
                    f"{var.space!r}, and the strong second derivative of a P1 field is degenerate (§1.11.9). "
                    f"Raise the variable's space or use the weak-form escape hatch.",
                )

    def _check_param_use_site(self, nm: str, ctx: _ExprContext) -> None:
        p = self._param_by_name.get(nm)
        if isinstance(p, ParameterExpression) and p.subdomain is not None and p.subdomain != ctx.eval_subdomain:
            self._error(
                ctx.path,
                f"parameter {nm!r} is scoped to subdomain {p.subdomain!r} (it uses geometric helpers) and cannot "
                f"be used from an expression evaluated on {ctx.eval_subdomain!r} (§1.11.10)",
            )

    def _check_weak_form_structure(self, eq: WeakFormEquation, tree: Expr, path: str) -> None:
        governed = eq.variable
        if not self._contains_name(tree, f"{governed}_test"):
            self._error(
                f"{path}.form",
                f"weak form must reference the test function {governed}_test at least once (§2.3.5)",
            )
        has_dt = self._contains_partial_t_of(tree, governed)
        if eq.temporality == "time_dependent" and not has_dt:
            self._error(f"{path}.form", f"time_dependent weak form must contain partial_t({governed}) (§1.9.5)")
        if eq.temporality == "steady_state" and has_dt:
            self._error(f"{path}.form", f"steady_state weak form must not contain partial_t({governed}) (§1.9.5)")

    def _check_parameter_cycles(self, trees: dict[str, Expr]) -> None:
        graph = {name: self._param_refs(tree) & set(trees) for name, tree in trees.items()}
        color: dict[str, int] = dict.fromkeys(graph, 0)  # 0=unvisited 1=on-stack 2=done
        stack: list[str] = []
        reported: set[frozenset[str]] = set()

        def dfs(n: str) -> None:
            color[n] = 1
            stack.append(n)
            for m in sorted(graph[n]):
                if color[m] == 1:
                    cycle = stack[stack.index(m) :]
                    if frozenset(cycle) not in reported:
                        reported.add(frozenset(cycle))
                        self._error("parameters", f"parameter expression cycle: {' -> '.join([*cycle, m])} (§1.11.3)")
                elif color[m] == 0:
                    dfs(m)
            stack.pop()
            color[n] = 2

        for n in sorted(graph):
            if color[n] == 0:
                dfs(n)

    # -- small AST / index helpers -------------------------------------------

    def _rank(self, subdomain_name: str | None) -> int:
        if subdomain_name is None:
            return -1
        s = self._subdomain_by_name.get(subdomain_name)
        return _KIND_RANK.get(s.kind, -1) if s is not None else -1

    def _children(self, node: Expr) -> tuple[Expr, ...]:
        if isinstance(node, Number | Name):
            return ()
        if isinstance(node, IndexAccess):
            return (node.base, node.index)
        if isinstance(node, UnaryOp):
            return (node.operand,)
        if isinstance(node, BinaryOp):
            return (node.left, node.right)
        if isinstance(node, FunctionCall):
            return node.args
        if isinstance(node, VectorLiteral):
            return node.components
        if isinstance(node, TensorLiteral):
            return node.rows
        assert_never(node)

    def _contains_name(self, node: Expr, name: str) -> bool:
        if isinstance(node, Name) and node.name == name:
            return True
        return any(self._contains_name(child, name) for child in self._children(node))

    def _contains_partial_t_of(self, node: Expr, var: str) -> bool:
        if (
            isinstance(node, FunctionCall)
            and node.callee == "partial_t"
            and any(self._contains_name(arg, var) for arg in node.args)
        ):
            return True
        return any(self._contains_partial_t_of(child, var) for child in self._children(node))

    def _uses_geometric_helper(self, node: Expr) -> bool:
        # Subdomain-relative geometry (`geom.normal`, `geom.radius`, …, but not the globally
        # available `geom.x`) requires a `subdomain:` scope where it appears (§2.2.3).
        if isinstance(node, Name) and node.name in SCOPED_GEOMETRY_NAMES:
            return True
        return any(self._uses_geometric_helper(child) for child in self._children(node))

    def _local_variable_names(self, node: Expr, subdomain: str | None) -> set[str]:
        found: set[str] = set()
        if isinstance(node, Name) and subdomain is not None and (node.name, subdomain) in self._var_by_key:
            found.add(node.name)
        for child in self._children(node):
            found |= self._local_variable_names(child, subdomain)
        return found

    def _param_refs(self, node: Expr) -> set[str]:
        found: set[str] = set()
        if isinstance(node, Name) and node.name in self._param_names:
            found.add(node.name)
        for child in self._children(node):
            found |= self._param_refs(child)
        return found

    # -- §1.11.5 type inference / type-checking ------------------------------

    def _check_expr_type(self, tree: Expr, ctx: _ExprContext, expected: frozenset[VariableType]) -> None:
        inferred = self._infer(tree, ctx)
        if inferred == "error" or inferred in expected:
            return
        # The numeric literal 0 is the zero of any type (the reference models
        # write `0` for a zero vector velocity / IC). Allow it without an
        # explicit `[0, 0]`; every other scalar-where-vector case is a no-broadcast error.
        if isinstance(tree, Number) and tree.value == 0.0:
            return
        self._error(
            ctx.path,
            f"expression has type {inferred}, expected {self._format_types(expected)} (§1.11.5)",
        )

    def _infer(self, node: Expr, ctx: _ExprContext) -> _Type:
        if isinstance(node, Number):
            return "scalar"
        if isinstance(node, Name):
            return self._name_type(node.name, ctx)
        if isinstance(node, UnaryOp):
            operand = self._infer(node.operand, ctx)
            if node.op == "!" and operand not in ("scalar", "error"):
                self._error(ctx.path, f"! requires a scalar operand, got {operand} (§1.11.5)")
                return "error"
            return operand
        if isinstance(node, BinaryOp):
            return self._infer_binary(node, ctx)
        if isinstance(node, IndexAccess):
            base_type = self._infer(node.base, ctx)
            index_type = self._infer(node.index, ctx)
            if index_type not in ("scalar", "error"):
                self._error(ctx.path, f"index must be scalar, got {index_type} (§1.11.5)")
            if base_type == "vector":
                return "scalar"
            if base_type == "symmetric_tensor":
                return "vector"
            if base_type == "error":
                return "error"
            self._error(ctx.path, f"cannot index into a {base_type} value (§1.11.5)")
            return "error"
        if isinstance(node, VectorLiteral):
            for component in node.components:
                component_type = self._infer(component, ctx)
                if component_type not in ("scalar", "error"):
                    self._error(ctx.path, f"vector literal components must be scalar, got {component_type} (§1.11.5)")
            return "vector"
        if isinstance(node, TensorLiteral):
            for row in node.rows:
                self._infer(row, ctx)
            return "symmetric_tensor"
        if isinstance(node, FunctionCall):
            return self._infer_call(node, ctx)
        assert_never(node)

    def _infer_binary(self, node: BinaryOp, ctx: _ExprContext) -> _Type:
        left = self._infer(node.left, ctx)
        right = self._infer(node.right, ctx)
        if left == "error" or right == "error":
            return "error"
        op = node.op
        if op in ("+", "-"):
            if left == right:
                return left
            self._error(
                ctx.path,
                f"cannot compute {left} {op} {right}; operands must match, and there is no implicit broadcast (§1.8.8)",
            )
            return "error"
        if op == "*":
            if left == "scalar":
                return right
            if right == "scalar":
                return left
            self._error(
                ctx.path,
                f"cannot multiply {left} by {right}; use inner(...) / outer(...) for products of non-scalars (§1.11.5)",
            )
            return "error"
        if op == "/":
            if right != "scalar":
                self._error(ctx.path, f"divisor must be scalar, got {right} (§1.11.5)")
                return "error"
            return left
        if op in ("<", ">", "<=", ">=", "==", "!=", "&&", "||"):
            # relational / logical operators — scalar operands, scalar (boolean-as-0/1) result.
            # Mainly used inside conditionals, e.g. `if(c > 0, a, b)`.
            if left != "scalar" or right != "scalar":
                self._error(ctx.path, f"{op} requires scalar operands, got {left} {op} {right} (§1.11.5)")
                return "error"
            return "scalar"
        if left == "scalar" and right == "scalar":  # op == "**"
            return "scalar"
        self._error(ctx.path, f"** requires scalar operands, got {left} ** {right} (§1.11.5)")
        return "error"

    def _infer_call(self, node: FunctionCall, ctx: _ExprContext) -> _Type:
        callee = node.callee
        args = node.args
        if callee == "trace":
            return self._trace_type(node, ctx)
        if callee in CALCULUS_OPERATORS:
            return self._calculus_type(callee, args, ctx)
        if callee == "if":
            if args:
                condition = self._infer(args[0], ctx)
                if condition not in ("scalar", "error"):
                    self._error(ctx.path, f"if(...) condition must be scalar, got {condition} (§1.11.5)")
            branches = [self._infer(a, ctx) for a in args[1:]]
            if len(branches) == 2 and "error" not in branches:
                if branches[0] == branches[1]:
                    return branches[0]
                self._error(
                    ctx.path,
                    f"if(cond, a, b) branches must match, got {branches[0]} and {branches[1]} (§1.11.5)",
                )
            return "error"
        if callee == "inner":
            types = [self._infer(a, ctx) for a in args]
            if len(types) == 2 and "error" not in types and types[0] != types[1]:
                self._error(ctx.path, f"inner(a, b) requires matching ranks, got {types[0]} and {types[1]} (§1.11.5)")
            return "scalar"
        if callee == "outer":
            for a in args:
                self._infer(a, ctx)
            return "symmetric_tensor"
        if callee == "cross":
            for a in args:
                self._infer(a, ctx)
            return "vector"
        if callee == "partial_t":
            return self._infer(args[0], ctx) if args else "error"
        if callee in STANDARD_FUNCTIONS:
            for a in args:
                arg_type = self._infer(a, ctx)
                if arg_type not in ("scalar", "error"):
                    self._error(ctx.path, f"{callee}(...) requires scalar arguments, got {arg_type} (§1.11.5)")
            return "scalar"
        return "error"  # unknown function or measure — already handled in the resolution walk

    def _calculus_type(self, callee: str, args: tuple[Expr, ...], ctx: _ExprContext) -> _Type:
        if len(args) != 1:
            for a in args:
                self._infer(a, ctx)
            self._error(ctx.path, f"{callee}(...) takes exactly one argument (§1.8.5)")
            return "error"
        arg_type = self._infer(args[0], ctx)
        if arg_type == "error":
            return "error"
        in_type, out_type = _CALCULUS_SIGNATURE[callee]
        if arg_type != in_type:
            self._error(ctx.path, f"{callee}(...) requires a {in_type} argument, got {arg_type} (§1.8.5)")
            return "error"
        return out_type

    def _trace_type(self, node: FunctionCall, ctx: _ExprContext) -> _Type:
        if len(node.args) != 1 or not isinstance(node.args[0], Name):
            return "error"
        nm = node.args[0].name
        for host in sorted(self._var_subdomains.get(nm, set())):
            if self._rank(host) > self._rank(ctx.eval_subdomain):
                var = self._var_by_key.get((nm, host))
                if var is not None:
                    return var.type
        return "error"

    def _name_type(self, nm: str, ctx: _ExprContext) -> _Type:
        s = ctx.eval_subdomain
        if ctx.kind == "weak_form":
            if nm in MEASURES:
                return "error"
            if nm.endswith("_test"):
                base = nm[: -len("_test")]
                test_var = self._var_by_key.get((base, s)) if s is not None else None
                return test_var.type if test_var is not None else "error"
        if s is not None:
            local = self._var_by_key.get((nm, s))
            if local is not None:
                return local.type
        if nm in self._param_names:
            return self._param_type(nm)
        if "." in nm:  # a namespaced built-in: geom.* / sim.* (ADR 006)
            return self._qualified_type(nm)
        return "error"

    def _qualified_type(self, nm: str) -> _Type:
        # geom.x, geom.normal, geom.tangent are vectors; the curvatures, radius, azimuth, and the
        # sim.* quantities are scalars. Unknown members are already diagnosed in the resolution walk.
        if nm in ("geom.x", "geom.normal", "geom.tangent"):
            return "vector"
        if nm in QUALIFIED_BUILTINS:
            return "scalar"
        return "error"

    def _param_type(self, nm: str) -> _Type:
        p = self._param_by_name.get(nm)
        if isinstance(p, ParameterExpression):
            return p.type
        if isinstance(p, ParameterConstant | ParameterRegionMap):
            return "scalar"
        return "error"

    def _format_types(self, types: frozenset[VariableType]) -> str:
        return " or ".join(sorted(types))
