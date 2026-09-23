"""Compile a formalism expression AST into a UFL expression.

This is the `AST → UFL` stage of the pipeline (ADR 004). It walks the syntactic
AST from `formalism.expr` and emits UFL, resolving each `Name` through a
`CompileContext` whose symbol table maps formalism names to concrete UFL objects
(parameters → `fem.Constant`, the namespaced built-ins `geom.x` → `SpatialCoordinate` and
`sim.t` → time Constant, variables → the solution `Function`, and calculus operators).

UFL is itself the expression IR; there is no second representation here. The
compiler assumes the expression has already passed validation (`formalism.validator`),
so an unresolved name or unsupported construct is an internal error, raised as
`CompileError` rather than returned as a diagnostic.

Supported so far: numeric literals, name lookups (parameters, the namespaced built-ins
`geom.x` → SpatialCoordinate and `sim.t` → time Constant, and — in a weak form — variables,
test functions, and measures the assembler binds), arithmetic (`+ - * / **`, unary `±`),
coordinate indexing (`geom.x[0]`), the standard scalar math functions, the geometry quantities
`geom.radius`/`geom.azimuth` (sugar for functions of position) and `geom.normal`/
`geom.mean_curvature` (bound by a mechanics solve), `trace(·)` (§1.8.2), **vector literals**
`[a, b]`, **tensor-algebra** (`inner`/`dot`/`outer`/`cross`), **first-order calculus**
(`grad`/`div`/`lapl` and the `_surf`/`_beltrami` variants, which on a codim-1 submesh are the
same UFL calls), and **`partial_t(u)`** (resolved to the time-discretised derivative the
assembler bound — the weak-form escape hatch). The `geom.tangent`/`geom.curvature1`/
`geom.curvature2` quantities, multi-argument standard functions, and tensor literals raise
`CompileError` until the increments that add them.

`trace(u)` compiles to `u`'s UFL object unchanged: the cross-dimensional *restriction*
of a bulk variable onto a lower-dimensional evaluation domain is realised by native
mixed-dimensional assembly (entity maps relating the submeshes; §1.8.2 "trace
evaluation is a backend concern"), not by a UFL wrapper. The compiler's only job is
to resolve the name; the assembler that builds a cross-subdomain form provides the
bulk variable's `Function` in the symbol table and the entity map at `fem.form` time.
The validator has already enforced that the argument is a single declared variable on
a strictly higher-dimensional subdomain (the direction rule), so the compiler trusts that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Name, Number, UnaryOp, VectorLiteral

# Formalism standard functions that map to a single-argument UFL function
# (§1.8.5). Multi-argument functions (atan2, min, max, pow) and the
# conditionals (if, step, sign) are added when a model needs them.
_UFL_UNARY_FUNCTIONS: dict[str, Any] = {
    "sin": ufl.sin,
    "cos": ufl.cos,
    "tan": ufl.tan,
    "asin": ufl.asin,
    "acos": ufl.acos,
    "atan": ufl.atan,
    "sinh": ufl.sinh,
    "cosh": ufl.cosh,
    "tanh": ufl.tanh,
    "exp": ufl.exp,
    "log": ufl.ln,  # the formalism `log` is the natural log; UFL spells it `ln`
    "log10": lambda a: ufl.ln(a) / ufl.ln(10.0),
    "sqrt": ufl.sqrt,
    "abs": abs,
}
# `floor` / `ceil` are accepted by the validator (they appear in real VCell models) but are
# non-differentiable, so UFL/DOLFINx has no operator for them — compiling one raises below.

# Two-argument tensor-algebra operators (§1.8, TENSOR_ALGEBRA).
_UFL_BINARY: dict[str, Any] = {"inner": ufl.inner, "dot": ufl.dot, "outer": ufl.outer, "cross": ufl.cross}

# Two-argument scalar math functions (§1.8.5). VCell emits `pow(a, b)` for exponentiation (the
# expression importer keeps it as a call) and `min` / `max` for clamping; these are the value-level
# counterparts of the `**` operator and the tensor `min`/`max` are not these.
_UFL_BINARY_MATH: dict[str, Any] = {
    "pow": lambda a, b: a**b,
    "atan2": ufl.atan2,
    "min": ufl.min_value,
    "max": ufl.max_value,
}

# First-order calculus operators (§1.8). On a codim-1 submesh `ufl.grad` / `ufl.div`
# are already the tangential (surface) operators, so the `_surf` variants are the
# same UFL calls — the distinction is the mesh, not the operator.
_UFL_CALCULUS: dict[str, Any] = {"grad": ufl.grad, "div": ufl.div, "grad_surf": ufl.grad, "div_surf": ufl.div}

# Random-variable primitives (IC-only): `(generator, p, q, n) -> n samples`. Realized once into a
# stored field so the draw is a fixed function of space (see `_compile_random`).
_RANDOM_SAMPLERS: dict[str, Any] = {
    "normal": lambda rng, mean, std, n: rng.normal(mean, std, n),
    "uniform": lambda rng, lo, hi, n: rng.uniform(lo, hi, n),
}


def _as_value(compiled: Any) -> Any:
    """Coerce a relational/logical result (a UFL `Condition`) to its 0/1 numeric value, so a
    boolean used arithmetically follows VCell semantics — `10*(x<5)` == `if(x<5, 10, 0)`.
    Non-conditions pass through unchanged."""
    return ufl.conditional(compiled, 1.0, 0.0) if isinstance(compiled, ufl.classes.Condition) else compiled


def _as_condition(compiled: Any) -> Any:
    """Coerce a numeric value to a UFL `Condition` (`x != 0`) where a boolean is required
    (logical operators, an `if(...)` condition). Conditions pass through unchanged."""
    return compiled if isinstance(compiled, ufl.classes.Condition) else ufl.ne(compiled, 0.0)


class CompileError(Exception):
    """A MathDescription expression could not be compiled to UFL. Indicates an
    unresolved name or a construct outside the current backend subset — both of
    which a validated MathDescription within the supported subset should avoid."""


@dataclass
class CompileContext:
    """The symbol environment for compilation. `mesh` is needed to build
    `fem.Constant`s; `symbols` maps formalism names to UFL objects (parameters as
    Constants, variables/test functions/measures in a weak-form context).
    `time_derivatives` maps a variable name to the UFL object `partial_t(<var>)`
    compiles to — the backend's time-discretised derivative (e.g. `(uⁿ⁺¹ − uⁿ)/dt`
    for backward Euler) — populated only for a time-dependent weak form.

    `rng` is the seeded generator the random IC primitives (`normal`/`uniform`) draw
    from — once each, into a stored field — so a model's random initial conditions are
    reproducible (one generator per compile, drawn in compile order); set its seed via
    `_compile_context(..., seed=...)`."""

    mesh: Mesh
    symbols: dict[str, UflExpr] = field(default_factory=dict)
    time_derivatives: dict[str, UflExpr] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def compile_expression(node: Expr, ctx: CompileContext) -> UflExpr:
    """Compile one expression AST node to a UFL expression."""

    if isinstance(node, Number):
        # petsc4py stubs type PETSc.ScalarType as a non-callable dtype; it is a
        # callable scalar alias at runtime.
        return fem.Constant(ctx.mesh, PETSc.ScalarType(node.value))  # type: ignore[operator]
    if isinstance(node, Name):
        if "." in node.name:
            # a namespaced built-in (geom.* / sim.*, ADR 006) — or a parameter whose VCell name has a dot
            # (VCell names spatial-process quantities that way, e.g. ``vproc_1.velocityX``)
            bound = ctx.symbols.get(node.name)
            return bound if bound is not None else _resolve_qualified(node.name, ctx)
        try:
            return ctx.symbols[node.name]
        except KeyError:
            raise CompileError(f"unresolved name {node.name!r} (not in the compile context)") from None
    if isinstance(node, UnaryOp):
        operand = compile_expression(node.operand, ctx)
        if node.op == "+":
            return _as_value(operand)
        if node.op == "!":
            return ufl.Not(_as_condition(operand))
        return -_as_value(operand)
    if isinstance(node, BinaryOp):
        left = compile_expression(node.left, ctx)
        right = compile_expression(node.right, ctx)
        # Arithmetic and relational operators take *values*: a relational/logical operand
        # (a UFL Condition) is coerced to its 0/1 number first, so `10*(x<5)` means
        # `10 if x<5 else 0` (VCell boolean-as-number semantics). Logical operators take
        # *conditions* (a numeric operand becomes `!= 0`).
        lv, rv = _as_value(left), _as_value(right)
        match node.op:
            case "+":
                return lv + rv
            case "-":
                return lv - rv
            case "*":
                return lv * rv
            case "/":
                return lv / rv
            case "**":
                return lv**rv
            case "<":
                return ufl.lt(lv, rv)
            case ">":
                return ufl.gt(lv, rv)
            case "<=":
                return ufl.le(lv, rv)
            case ">=":
                return ufl.ge(lv, rv)
            case "==":
                return ufl.eq(lv, rv)
            case "!=":
                return ufl.ne(lv, rv)
            case "&&":
                return ufl.And(_as_condition(left), _as_condition(right))
            case "||":
                return ufl.Or(_as_condition(left), _as_condition(right))
    if isinstance(node, IndexAccess):
        base = compile_expression(node.base, ctx)
        if not isinstance(node.index, Number) or not node.index.value.is_integer():
            raise CompileError("index must be an integer literal, e.g. x[0]")
        return base[int(node.index.value)]
    if isinstance(node, VectorLiteral):
        return ufl.as_vector([_as_value(compile_expression(c, ctx)) for c in node.components])
    if isinstance(node, FunctionCall):
        return _compile_call(node, ctx)
    raise CompileError(f"{type(node).__name__} is not supported by the backend yet")


def _resolve_qualified(name: str, ctx: CompileContext) -> UflExpr:
    """Resolve a namespaced built-in quantity `geom.*` / `sim.*` to its UFL object (ADR 006).

    `geom.x` (position) and `sim.t` (time) are bound directly in the symbol table by the
    assembler; the curvilinear `geom.radius` / `geom.azimuth` are sugar for functions of the
    position field; `geom.normal` / `geom.mean_curvature` resolve to the projected surface normal
    and mean curvature a mechanics assembler binds (on a discrete membrane these are not
    pointwise-defined — they come from a weak projection κ = H·n — so they are unavailable
    outside a force-balance solve)."""

    root, _, member = name.partition(".")
    if root == "geom":
        if member == "x":
            return ctx.symbols["geom.x"]
        if member == "radius":
            x = ctx.symbols["geom.x"]
            return ufl.sqrt(ufl.dot(x, x))
        if member == "azimuth":
            x = ctx.symbols["geom.x"]
            return ufl.atan2(x[1], x[0])
        if member in ("normal", "mean_curvature"):
            bound = ctx.symbols.get("__normal__" if member == "normal" else "__mean_curvature__")
            if bound is None:
                quantity = "normal" if member == "normal" else "mean curvature"
                raise CompileError(
                    f"geom.{member} (surface {quantity}) is only available where the curvature projection is "
                    f"bound — i.e. in a mechanics (force-balance) solve"
                )
            return bound
        raise CompileError(f"geom.{member} is not supported by the backend yet")
    if root == "sim":
        if member == "t":
            return ctx.symbols["sim.t"]
        raise CompileError(f"sim.{member} is not supported by the backend yet")
    raise CompileError(f"unresolved name {name!r} (not in the compile context)")


def _compile_call(node: FunctionCall, ctx: CompileContext) -> UflExpr:
    # `partial_t(<var>)` resolves to the backend's time-discretised derivative, so it
    # is looked up by *name* (not compiled as an expression) — only valid in a
    # time-dependent weak form where the assembler bound it (§1.5.4).
    if node.callee == "partial_t":
        if len(node.args) != 1 or not isinstance(node.args[0], Name):
            raise CompileError("partial_t(...) takes a single variable name")
        derivative = ctx.time_derivatives.get(node.args[0].name)
        if derivative is None:
            raise CompileError(
                f"partial_t({node.args[0].name}) has no time derivative bound — it is only valid for a variable "
                f"governed by a time-dependent weak-form equation"
            )
        return derivative

    args = [compile_expression(arg, ctx) for arg in node.args]
    if node.callee == "if":
        # `if(condition, then, else)` → a UFL conditional. The condition is coerced to a
        # boolean (a bare numeric condition becomes `!= 0`); the branches to values.
        if len(args) != 3:
            raise CompileError("if(condition, then, else) takes exactly three arguments")
        return ufl.conditional(_as_condition(args[0]), _as_value(args[1]), _as_value(args[2]))
    if node.callee == "trace":
        # The trace of a higher-dimensional variable onto a lower-dimensional
        # evaluation domain (§1.8.2). At the UFL level this is the variable itself;
        # the actual restriction is mixed-dimensional assembly's job (entity maps),
        # so the compiler just returns the resolved argument. The validator has
        # guaranteed a single variable-name argument crossing high → low dimension.
        if len(args) != 1:
            raise CompileError("trace(...) takes exactly one argument")
        return args[0]
    binary = _UFL_BINARY.get(node.callee)
    if binary is not None:
        if len(args) != 2:
            raise CompileError(f"{node.callee}(...) takes two arguments")
        return binary(args[0], args[1])
    calculus = _UFL_CALCULUS.get(node.callee)
    if calculus is not None:
        return calculus(args[0])
    if node.callee in ("lapl", "lapl_beltrami"):
        return ufl.div(ufl.grad(args[0]))  # ∇·∇ — Beltrami on a submesh
    binary_math = _UFL_BINARY_MATH.get(node.callee)
    if binary_math is not None:
        if len(args) != 2:
            raise CompileError(f"{node.callee}(...) takes two arguments")
        return binary_math(*[_as_value(a) for a in args])
    if node.callee in _RANDOM_SAMPLERS:
        return _compile_random(node.callee, args, ctx)
    fn = _UFL_UNARY_FUNCTIONS.get(node.callee)
    if fn is not None:
        return fn(*[_as_value(a) for a in args])
    raise CompileError(f"function {node.callee!r} is not supported by the backend yet")


def _compile_random(callee: str, args: list[Any], ctx: CompileContext) -> UflExpr:
    """Realize a random IC primitive (`normal(mean, std)` / `uniform(lo, hi)`) into a **stored** P1
    field: draw one sample per DOF from `ctx.rng`, fill a `Function`, and return it as a UFL coefficient.
    Drawing once (not per evaluation) is what makes the noise a fixed function of space — the same point
    returns the same value under any later substitution/assembly. The distribution parameters must be
    constants in v1 (a spatially varying scale is a later increment)."""

    if len(args) != 2:
        raise CompileError(f"{callee}(...) takes two arguments (the distribution parameters)")
    params = []
    for a in args:
        if not isinstance(a, fem.Constant):
            raise CompileError(f"{callee}(...) parameters must be constants in v1, got a non-constant expression")
        params.append(float(a.value))
    field_function = fem.Function(fem.functionspace(ctx.mesh, ("Lagrange", 1)))
    field_function.x.array[:] = _RANDOM_SAMPLERS[callee](ctx.rng, params[0], params[1], field_function.x.array.size)
    field_function.x.scatter_forward()
    return field_function
