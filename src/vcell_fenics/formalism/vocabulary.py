"""The closed reserved-name vocabulary of the expression language.

These are the names a MathDescription's expressions may invoke but a user may
*not* reuse as a variable, parameter, or subdomain name (§2.4.1 — shadowing is
a construction-time error). The sets mirror docs/modeling/declarative-formalism.md
§1.8.3–§1.8.5 and §2.3.4.

This module currently exports the *name sets* the shadowing check needs. Arg
counts, argument/return types, and domain constraints (e.g. `grad_surf` is
surface-only, `H` is codim-1-only) are added when expression type-checking and
the operator-usage rules land — they are not needed to detect shadowing.
"""

from __future__ import annotations

# Time and spatial-coordinate accessors (§1.8.3).
COORDINATES_AND_TIME: frozenset[str] = frozenset({"t", "x"})

# Elementary / transcendental / piecewise functions (§1.8.5).
STANDARD_FUNCTIONS: frozenset[str] = frozenset(
    {
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "exp", "log", "sqrt", "abs", "min", "max", "pow",
        "if", "step", "sign",
    }
)  # fmt: skip

# Geometric helpers (§1.8.4).
GEOMETRIC_HELPERS: frozenset[str] = frozenset({"n", "H", "kappa1", "kappa2", "tangent", "theta", "phi", "r"})

# Calculus operators on variables (§1.8.5).
CALCULUS_OPERATORS: frozenset[str] = frozenset({"grad", "div", "lapl", "grad_surf", "div_surf", "lapl_beltrami"})

# Cross-dimensional reference (§1.8.2) and tensor algebra (§2.3.4).
TRACE: frozenset[str] = frozenset({"trace"})
TENSOR_ALGEBRA: frozenset[str] = frozenset({"inner", "outer", "cross"})

# Time derivative, valid only in weak-form residuals (§2.3.4).
TIME_DERIVATIVE: frozenset[str] = frozenset({"partial_t"})

# Integration measures, valid only in weak-form residuals (§2.3.4). The
# parametrised forms (`ds(<boundary>)`, …) share these base names.
MEASURES: frozenset[str] = frozenset({"dx", "dx_Gamma", "dl", "dp", "ds", "dS", "dl_Gamma"})

# Every reserved name. A variable, parameter, or subdomain whose name is in
# this set is a shadowing error (§1.11.3, §2.4.1).
RESERVED_NAMES: frozenset[str] = (
    COORDINATES_AND_TIME
    | STANDARD_FUNCTIONS
    | GEOMETRIC_HELPERS
    | CALCULUS_OPERATORS
    | TRACE
    | TENSOR_ALGEBRA
    | TIME_DERIVATIVE
    | MEASURES
)
