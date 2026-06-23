"""The reserved-name vocabulary of the expression language (ADR 006).

Two disjoint kinds of name live here:

- **Built-in quantities** are namespaced under the roots ``geom`` and ``sim`` —
  ``geom.x``, ``geom.normal``, ``sim.t`` — and addressed as qualified names, never
  bare. The bare value namespace therefore belongs entirely to the user: a
  modeller may name a variable or parameter ``x``, ``t``, ``phi``, ``r``, … without
  colliding with a built-in.
- **Operators, functions, and measures** (``grad``, ``sin``, ``inner``, ``dx``, …)
  are reserved only in *call / measure position* (``name(...)`` or ``* dx``). They
  do not consume value-namespace names — a value is read as ``name``, a call as
  ``name(`` — so they never restrict user variable names.

Consequently the only names a user may *not* take (`RESERVED_NAMES`) are the two
namespace roots and the measures (which read as bare tokens inside weak forms).
The sets mirror docs/modeling/declarative-formalism.md §1.8.3–§1.8.5 and §2.3.4.
"""

from __future__ import annotations

# -- Built-in quantities, addressed as qualified names (ADR 006) ------------

# `geom.x` is the position field, available everywhere (the old bare `x`). The rest are
# *subdomain-relative* geometry — the boundary/surface normal and curvatures, the tangent, and
# the curvilinear radius/azimuth (defined relative to the geometry) — so a parameter expression
# that uses one must declare a `subdomain:` scope (§2.2.3).
GEOMETRY_POSITION: str = "x"
GEOMETRY_SCOPED_MEMBERS: frozenset[str] = frozenset(
    {"normal", "mean_curvature", "curvature1", "curvature2", "tangent", "radius", "azimuth"}
)
GEOMETRY_MEMBERS: frozenset[str] = GEOMETRY_SCOPED_MEMBERS | {GEOMETRY_POSITION}

# `sim.t` is the simulation time (the old bare `t`); `sim.dt` the step.
SIMULATION_MEMBERS: frozenset[str] = frozenset({"t", "dt"})

# The bare namespace roots. A qualified built-in is `root.member`.
NAMESPACE_ROOTS: frozenset[str] = frozenset({"geom", "sim"})

# Fully-qualified built-in names, e.g. `geom.x`, `geom.normal`, `sim.t`.
GEOMETRY_NAMES: frozenset[str] = frozenset(f"geom.{m}" for m in GEOMETRY_MEMBERS)
SCOPED_GEOMETRY_NAMES: frozenset[str] = frozenset(f"geom.{m}" for m in GEOMETRY_SCOPED_MEMBERS)
SIMULATION_NAMES: frozenset[str] = frozenset(f"sim.{m}" for m in SIMULATION_MEMBERS)
QUALIFIED_BUILTINS: frozenset[str] = GEOMETRY_NAMES | SIMULATION_NAMES

# -- Operators / functions / measures (call or measure position only) -------

# Elementary / transcendental / piecewise functions (§1.8.5).
STANDARD_FUNCTIONS: frozenset[str] = frozenset(
    {
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "exp", "log", "log10", "sqrt", "abs", "min", "max", "pow",
        "floor", "ceil",
        "if", "step", "sign",
    }
)  # fmt: skip

# Calculus operators on variables (§1.8.5).
CALCULUS_OPERATORS: frozenset[str] = frozenset({"grad", "div", "lapl", "grad_surf", "div_surf", "lapl_beltrami"})

# Cross-dimensional reference (§1.8.2) and tensor algebra (§2.3.4).
TRACE: frozenset[str] = frozenset({"trace"})
TENSOR_ALGEBRA: frozenset[str] = frozenset({"inner", "outer", "cross"})

# Random-variable primitives, valid only in an initial condition. `normal(mean, std)` and
# `uniform(lo, hi)` draw a per-DOF sample; each is realized ONCE into a fixed (seeded) field, so the
# expression stays a pure function of space — a fresh draw on every evaluation would not be (the value
# at a point would change under re-assembly/substitution), which is why they are confined to the IC
# (realized exactly once). Spinodal noise is then just `mean + normal(0, amplitude)`.
RANDOM_FUNCTIONS: frozenset[str] = frozenset({"normal", "uniform"})

# Time derivative, valid only in weak-form residuals (§2.3.4).
TIME_DERIVATIVE: frozenset[str] = frozenset({"partial_t"})

# Integration measures, valid only in weak-form residuals (§2.3.4). The
# parametrised forms (`ds(<boundary>)`, …) share these base names.
MEASURES: frozenset[str] = frozenset({"dx", "dx_Gamma", "dl", "dp", "ds", "dS", "dl_Gamma"})

# Names that may appear only as a call's callee, never as a bare value.
RESERVED_CALLABLES: frozenset[str] = (
    STANDARD_FUNCTIONS | CALCULUS_OPERATORS | TRACE | TENSOR_ALGEBRA | RANDOM_FUNCTIONS | TIME_DERIVATIVE
)

# Names a user may NOT take for a subdomain, variable, or parameter (§1.11.3, §2.4.1). Per ADR 006
# the bare value namespace is the user's: only the two namespace roots and the measures (bare
# tokens in weak forms) are off-limits. Operators/functions are *not* here — they never collide
# with a value, so a user may name a variable `grad` if they insist (and call `grad(...)` too).
RESERVED_NAMES: frozenset[str] = NAMESPACE_ROOTS | MEASURES
