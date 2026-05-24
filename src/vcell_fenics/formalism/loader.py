"""YAML / JSON → MathDescription loader.

Parses the surface form documented in §2.1 / §2.2 of
docs/modeling/declarative-formalism.md into the dataclass schema. The
loader handles structural concerns only:

- Top-level `math_description:` envelope unwrapping.
- Tag dispatch for discriminated unions (Motion, Parameter, Equation,
  BoundaryCondition) based on `kind` field and presence of distinguishing
  sub-fields (e.g. `expression` vs `value` for Parameter).
- Required-field presence checks and unknown-field rejection.
- Default substitution where the schema declares defaults.

Semantic validation (name resolution, type checking, temporality
consistency, etc.) belongs to the validator module and is intentionally
not done here. The loader's contract is "structurally well-formed YAML/JSON
→ structurally well-formed dataclass tree."

Errors are raised as `FormalismLoadError` carrying a JSON-pointer-style
path string indicating where in the document the problem occurred, so
debugging large models is tractable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    BCNeumann,
    BCRobin,
    BoundaryCondition,
    Equation,
    MathDescription,
    Motion,
    MotionNone,
    MotionPrescribedDisplacement,
    MotionPrescribedVelocity,
    MotionUnknown,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    ParameterRegionMap,
    Subdomain,
    SubdomainKind,
    TemplateEquation,
    Temporality,
    Variable,
    VariableType,
    WeakFormEquation,
)


class FormalismLoadError(Exception):
    """A MathDescription document is structurally malformed.

    `path` is a JSON-pointer-style string indicating the offending field
    in the source document (e.g. `math_description.parameters[3].kind`).
    `message` is the human-readable explanation. The combined str()
    presentation reads "<path>: <message>".
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def load_yaml(source: str | Path) -> MathDescription:
    """Parse a YAML document or file into a MathDescription.

    `source` is either YAML text (str) or a file path. A heuristic
    distinguishes them: anything that contains a newline or has no
    plausible filesystem path is treated as text; everything else is
    opened as a file. To force the file-path interpretation, wrap the
    string in `pathlib.Path`.
    """

    text = _read_text(source)
    raw = yaml.safe_load(text)
    if raw is None:
        raise FormalismLoadError("", "empty document")
    if not isinstance(raw, dict):
        raise FormalismLoadError("", f"expected a mapping at the top level, got {type(raw).__name__}")
    return load_dict(raw)


def load_json(source: str | Path) -> MathDescription:
    """Parse a JSON document or file into a MathDescription.

    Same source-interpretation heuristic as :func:`load_yaml`.
    """

    text = _read_text(source)
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise FormalismLoadError("", f"expected a mapping at the top level, got {type(raw).__name__}")
    return load_dict(raw)


def load_dict(raw: dict[str, Any]) -> MathDescription:
    """Parse an already-deserialised dict (e.g. from YAML/JSON) into a
    MathDescription. The dict must have the top-level
    `math_description:` envelope.
    """

    if "math_description" not in raw:
        raise FormalismLoadError(
            "",
            "missing top-level 'math_description:' envelope (the doc's §2.1.2 schema)",
        )
    body = raw["math_description"]
    if not isinstance(body, dict):
        raise FormalismLoadError(
            "math_description",
            f"must be a mapping, got {type(body).__name__}",
        )
    return _parse_math_description(body, "math_description")


# ---------------------------------------------------------------------------
# Per-entity parsers. Each takes the raw dict and the path-so-far for
# error reporting; each returns a dataclass instance.
# ---------------------------------------------------------------------------


_MD_KEYS = {"geometry", "subdomains", "variables", "equations", "parameters", "boundary_conditions"}


def _parse_math_description(d: dict[str, Any], path: str) -> MathDescription:
    _require_keys(d, ["geometry", "subdomains", "variables", "equations"], path)
    _reject_unknown(d, _MD_KEYS, path)
    geometry = _str(d, "geometry", path)

    subdomains = [
        _parse_subdomain(sub, f"{path}.subdomains[{i}]") for i, sub in enumerate(_list(d, "subdomains", path))
    ]
    if not subdomains:
        raise FormalismLoadError(f"{path}.subdomains", "must be non-empty")

    variables = [_parse_variable(v, f"{path}.variables[{i}]") for i, v in enumerate(_list(d, "variables", path))]
    if not variables:
        raise FormalismLoadError(f"{path}.variables", "must be non-empty")

    equations = [_parse_equation(e, f"{path}.equations[{i}]") for i, e in enumerate(_list(d, "equations", path))]
    if not equations:
        raise FormalismLoadError(f"{path}.equations", "must be non-empty")

    parameters = [_parse_parameter(p, f"{path}.parameters[{i}]") for i, p in enumerate(d.get("parameters") or [])]
    boundary_conditions = [
        _parse_boundary_condition(b, f"{path}.boundary_conditions[{i}]")
        for i, b in enumerate(d.get("boundary_conditions") or [])
    ]

    return MathDescription(
        geometry=geometry,
        subdomains=subdomains,
        variables=variables,
        equations=equations,
        parameters=parameters,
        boundary_conditions=boundary_conditions,
    )


_SUBDOMAIN_KEYS = {"name", "kind", "motion"}
_SUBDOMAIN_KINDS = {"volume", "surface", "curve", "point"}


def _parse_subdomain(d: dict[str, Any], path: str) -> Subdomain:
    _require_keys(d, ["name", "kind"], path)
    _reject_unknown(d, _SUBDOMAIN_KEYS, path)
    kind = _enum(d, "kind", path, _SUBDOMAIN_KINDS)
    motion_raw = d.get("motion")
    motion: Motion = _parse_motion(motion_raw, f"{path}.motion") if motion_raw is not None else MotionNone()
    # The enum check above guarantees kind is in _SUBDOMAIN_KINDS, hence one of
    # the four SubdomainKind literals; cast for mypy without weakening the API.
    return Subdomain(name=_str(d, "name", path), kind=_cast_subdomain_kind(kind), motion=motion)


_MOTION_KINDS = {"none", "prescribed", "unknown"}


def _parse_motion(d: Any, path: str) -> Motion:
    if not isinstance(d, dict):
        raise FormalismLoadError(path, f"must be a mapping, got {type(d).__name__}")
    kind = _enum(d, "kind", path, _MOTION_KINDS)
    if kind == "none":
        _reject_unknown(d, {"kind"}, path)
        return MotionNone()
    if kind == "prescribed":
        _reject_unknown(d, {"kind", "velocity", "displacement"}, path)
        has_v = "velocity" in d
        has_d = "displacement" in d
        if has_v and has_d:
            raise FormalismLoadError(
                path,
                "prescribed motion must declare exactly one of 'velocity' or 'displacement', not both",
            )
        if not (has_v or has_d):
            raise FormalismLoadError(
                path,
                "prescribed motion requires either 'velocity' or 'displacement'",
            )
        if has_v:
            return MotionPrescribedVelocity(velocity=_str(d, "velocity", path))
        return MotionPrescribedDisplacement(displacement=_str(d, "displacement", path))
    # kind == "unknown" by enum check.
    _reject_unknown(d, {"kind", "variable"}, path)
    return MotionUnknown(variable=_str(d, "variable", path))


_VARIABLE_KEYS = {"name", "subdomain", "type", "space"}
_VARIABLE_TYPES = {"scalar", "vector", "symmetric_tensor"}


def _parse_variable(d: dict[str, Any], path: str) -> Variable:
    _require_keys(d, ["name", "subdomain"], path)
    _reject_unknown(d, _VARIABLE_KEYS, path)
    var_type_str = d.get("type", "scalar")
    if var_type_str not in _VARIABLE_TYPES:
        raise FormalismLoadError(
            f"{path}.type",
            f"unknown variable type {var_type_str!r} (expected one of: {sorted(_VARIABLE_TYPES)})",
        )
    return Variable(
        name=_str(d, "name", path),
        subdomain=_str(d, "subdomain", path),
        type=_cast_variable_type(var_type_str),
        space=str(d.get("space", "lagrange_p1")),
    )


# Parameter dispatch.
#
# The doc allows shorthand: `{name, value}` is sugar for
# `{name, kind: scalar, value}`. Dispatch order:
#   1. explicit `kind: region_map`        → ParameterRegionMap
#   2. `expression` field present         → ParameterExpression
#      (with optional explicit `kind: expression`)
#   3. `value` field present              → ParameterConstant
#      (with optional explicit `kind: scalar`)
#   4. otherwise                          → error
#
# Explicit `kind` values that conflict with the field shape are rejected
# (e.g. `kind: scalar` with `expression:` set).

_PARAM_CONSTANT_KEYS = {"kind", "name", "value"}
_PARAM_EXPRESSION_KEYS = {"kind", "name", "type", "subdomain", "expression"}
_PARAM_REGION_MAP_KEYS = {"kind", "name", "subdomain", "values"}


def _parse_parameter(d: dict[str, Any], path: str) -> Parameter:
    if not isinstance(d, dict):
        raise FormalismLoadError(path, f"must be a mapping, got {type(d).__name__}")
    if "name" not in d:
        raise FormalismLoadError(path, "missing required field 'name'")
    kind = d.get("kind")

    if kind == "region_map":
        _reject_unknown(d, _PARAM_REGION_MAP_KEYS, path)
        _require_keys(d, ["subdomain", "values"], path)
        values_raw = d["values"]
        if not isinstance(values_raw, dict):
            raise FormalismLoadError(
                f"{path}.values",
                f"must be a mapping of region_name → number, got {type(values_raw).__name__}",
            )
        values = {str(rk): _float(rv, f"{path}.values[{rk!r}]") for rk, rv in values_raw.items()}
        if not values:
            raise FormalismLoadError(f"{path}.values", "must contain at least one region entry")
        return ParameterRegionMap(
            name=_str(d, "name", path),
            subdomain=_str(d, "subdomain", path),
            values=values,
        )

    has_expression = "expression" in d
    has_value = "value" in d

    if has_expression and has_value:
        raise FormalismLoadError(
            path,
            "parameter cannot declare both 'expression' and 'value' (those are mutually exclusive forms)",
        )

    if has_expression:
        if kind not in (None, "expression"):
            raise FormalismLoadError(
                f"{path}.kind",
                f"expression-valued parameter cannot have kind={kind!r}; expected 'expression' or omitted",
            )
        _reject_unknown(d, _PARAM_EXPRESSION_KEYS, path)
        param_type_str = d.get("type", "scalar")
        if param_type_str not in _VARIABLE_TYPES:
            raise FormalismLoadError(
                f"{path}.type",
                f"unknown parameter type {param_type_str!r} (expected one of: {sorted(_VARIABLE_TYPES)})",
            )
        return ParameterExpression(
            name=_str(d, "name", path),
            type=_cast_variable_type(param_type_str),
            expression=_str(d, "expression", path),
            subdomain=d.get("subdomain"),
        )

    if has_value:
        if kind not in (None, "scalar"):
            raise FormalismLoadError(
                f"{path}.kind",
                f"constant parameter cannot have kind={kind!r}; expected 'scalar' or omitted",
            )
        _reject_unknown(d, _PARAM_CONSTANT_KEYS, path)
        return ParameterConstant(name=_str(d, "name", path), value=_float(d["value"], f"{path}.value"))

    raise FormalismLoadError(
        path,
        "parameter must have one of: 'value' (constant), 'expression' (expression-valued), "
        "or 'kind: region_map' with 'values' (region-keyed)",
    )


# Equation dispatch — TemplateEquation vs WeakFormEquation by `template` value.

_TEMPLATE_EQUATION_KEYS = {"template", "variable", "subdomain", "temporality", "terms", "initial_condition"}
_WEAK_FORM_EQUATION_KEYS = {"template", "variable", "subdomain", "temporality", "form", "initial_condition"}
_TEMPORALITIES = {"time_dependent", "steady_state"}


def _parse_equation(d: dict[str, Any], path: str) -> Equation:
    if not isinstance(d, dict):
        raise FormalismLoadError(path, f"must be a mapping, got {type(d).__name__}")
    _require_keys(d, ["template", "variable", "subdomain", "temporality"], path)
    template = _str(d, "template", path)
    temporality_str = _enum(d, "temporality", path, _TEMPORALITIES)
    temporality = _cast_temporality(temporality_str)
    variable = _str(d, "variable", path)
    subdomain = _str(d, "subdomain", path)
    initial_condition = d.get("initial_condition")
    if initial_condition is not None:
        initial_condition = str(initial_condition)

    if template == "weak_form":
        _reject_unknown(d, _WEAK_FORM_EQUATION_KEYS, path)
        _require_keys(d, ["form"], path)
        return WeakFormEquation(
            variable=variable,
            subdomain=subdomain,
            temporality=temporality,
            form=_str(d, "form", path),
            initial_condition=initial_condition,
        )
    _reject_unknown(d, _TEMPLATE_EQUATION_KEYS, path)
    terms_raw = d.get("terms") or {}
    if not isinstance(terms_raw, dict):
        raise FormalismLoadError(
            f"{path}.terms",
            f"must be a mapping of slot_name → expression, got {type(terms_raw).__name__}",
        )
    terms = {str(k): str(v) for k, v in terms_raw.items()}
    return TemplateEquation(
        template=template,
        variable=variable,
        subdomain=subdomain,
        temporality=temporality,
        terms=terms,
        initial_condition=initial_condition,
    )


# BoundaryCondition dispatch by `kind`.

_BC_KINDS = {"dirichlet", "neumann", "robin", "interface_value_equality", "interface_flux_balance"}
_BC_DIRICHLET_KEYS = {"kind", "variable", "boundary", "expression"}
_BC_NEUMANN_KEYS = _BC_DIRICHLET_KEYS
_BC_ROBIN_KEYS = {"kind", "variable", "boundary", "alpha", "beta", "expression"}
_BC_IVE_KEYS = {"kind", "variable", "partner_variable", "boundary", "expression"}
_BC_IFB_KEYS = _BC_IVE_KEYS


def _parse_boundary_condition(d: dict[str, Any], path: str) -> BoundaryCondition:
    if not isinstance(d, dict):
        raise FormalismLoadError(path, f"must be a mapping, got {type(d).__name__}")
    kind = _enum(d, "kind", path, _BC_KINDS)
    if kind == "dirichlet":
        _reject_unknown(d, _BC_DIRICHLET_KEYS, path)
        _require_keys(d, ["variable", "boundary", "expression"], path)
        return BCDirichlet(
            variable=_str(d, "variable", path),
            boundary=_str(d, "boundary", path),
            expression=_str(d, "expression", path),
        )
    if kind == "neumann":
        _reject_unknown(d, _BC_NEUMANN_KEYS, path)
        _require_keys(d, ["variable", "boundary", "expression"], path)
        return BCNeumann(
            variable=_str(d, "variable", path),
            boundary=_str(d, "boundary", path),
            expression=_str(d, "expression", path),
        )
    if kind == "robin":
        _reject_unknown(d, _BC_ROBIN_KEYS, path)
        _require_keys(d, ["variable", "boundary", "alpha", "beta", "expression"], path)
        return BCRobin(
            variable=_str(d, "variable", path),
            boundary=_str(d, "boundary", path),
            alpha=_str(d, "alpha", path),
            beta=_str(d, "beta", path),
            expression=_str(d, "expression", path),
        )
    if kind == "interface_value_equality":
        _reject_unknown(d, _BC_IVE_KEYS, path)
        _require_keys(d, ["variable", "partner_variable", "boundary"], path)
        expression_raw = d.get("expression", "1")
        return BCInterfaceValueEquality(
            variable=_str(d, "variable", path),
            partner_variable=_str(d, "partner_variable", path),
            boundary=_str(d, "boundary", path),
            expression=str(expression_raw),
        )
    # kind == "interface_flux_balance" by the enum check.
    _reject_unknown(d, _BC_IFB_KEYS, path)
    _require_keys(d, ["variable", "partner_variable", "boundary", "expression"], path)
    return BCInterfaceFluxBalance(
        variable=_str(d, "variable", path),
        partner_variable=_str(d, "partner_variable", path),
        boundary=_str(d, "boundary", path),
        expression=_str(d, "expression", path),
    )


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _read_text(source: str | Path) -> str:
    """Read text from a file path or pass through a string.

    A `Path` always means file path. A `str` is treated as a path only
    if it has no newline AND a file exists at that location; otherwise
    it's treated as literal source text.
    """

    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if "\n" not in source:
        candidate = Path(source)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return source


def _require_keys(d: dict[str, Any], keys: list[str], path: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise FormalismLoadError(path, f"missing required field(s): {missing}")


def _reject_unknown(d: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise FormalismLoadError(
            path,
            f"unknown field(s) {unknown} (allowed: {sorted(allowed)})",
        )


def _str(d: dict[str, Any], key: str, path: str) -> str:
    v = d[key]
    if not isinstance(v, str):
        raise FormalismLoadError(f"{path}.{key}", f"must be a string, got {type(v).__name__}")
    return v


def _list(d: dict[str, Any], key: str, path: str) -> list[Any]:
    v = d[key]
    if not isinstance(v, list):
        raise FormalismLoadError(f"{path}.{key}", f"must be a list, got {type(v).__name__}")
    return v


def _float(v: Any, path: str) -> float:
    if isinstance(v, bool):  # bool is a subclass of int; reject explicitly
        raise FormalismLoadError(path, f"must be a number, got {type(v).__name__}")
    if isinstance(v, int | float):
        return float(v)
    raise FormalismLoadError(path, f"must be a number, got {type(v).__name__}")


def _enum(d: dict[str, Any], key: str, path: str, allowed: set[str]) -> str:
    if key not in d:
        raise FormalismLoadError(path, f"missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or v not in allowed:
        raise FormalismLoadError(
            f"{path}.{key}",
            f"got {v!r}; expected one of: {sorted(allowed)}",
        )
    return v


# mypy-friendly cast helpers: the Literal narrowing through `in _SET` checks
# isn't tight enough for mypy strict, so the loader narrows to str at the
# boundary, then converts to the Literal type via a typed cast. The runtime
# value is unchanged; only the static type narrows.


def _cast_subdomain_kind(v: str) -> SubdomainKind:
    # Caller guarantees v ∈ _SUBDOMAIN_KINDS via _enum().
    return v  # type: ignore[return-value]


def _cast_variable_type(v: str) -> VariableType:
    # Caller guarantees v ∈ _VARIABLE_TYPES.
    return v  # type: ignore[return-value]


def _cast_temporality(v: str) -> Temporality:
    # Caller guarantees v ∈ _TEMPORALITIES.
    return v  # type: ignore[return-value]
