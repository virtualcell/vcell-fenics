"""One cell moving in 2D: a cell force balance whose membrane moves, with no species (ADR 023; stage 4a).

The multi-compartment solver hands this path a T9 model whose one member is bounded by one moving membrane
(`integrate_multi_compartment` with ``motion_membrane``). The membrane is either

- **material:** it moves with the member's material, `motion: {kind: unknown, variable: v}` (ADR 018 decision 2), so
  its velocity is `trace(v)`; it may carry a tension (semi-implicit, ADR 018 decision 5) or an applied traction;
- **a growing front:** a prescribed velocity that reads `trace(v)`, the area and the normal (ADR 019 decision 5),
  with zero traction or held at `v = 0`, and no tension.

**The stride** (ADR 023 decision 3) is the explicit midpoint rule on the moving partition (`MovingPartition`,
ADRs 014–016). With `F(x)` the front's velocity on the geometry `x` (solve `v` on `x`, then the front law, with
`geom.normal` and `region_size` on `x`), a stride from `xⁿ` solves on `xⁿ` and moves half a stride by `F(xⁿ)` to
the midpoint mesh, then solves there and moves the other half by `2F(x^{n+½}) − F(xⁿ)`, so in total
`x^{n+1} = xⁿ + Δt·F(x^{n+½})`, second order in `Δt`. Each solve's semi-implicit tension uses the length of the
half-move it drives (decision 4); the sizes the front and the force balance read are re-measured after every move
(decision 5). Nothing is rebuilt after a move: with one member, the force balance lives on the member's submesh,
which the partition moves in place (decision 6). After a remesh the caller assembles all of it afresh on the new
partition and continues from ``t_start``.

**The scope** (decision 7) is checked by `moving_cell_scope`, which refuses by name what is outside it. A
stride interrupted by a remesh makes its second half on the new mesh by forward Euler from a fresh solve (the
first half's velocity is the old mesh's).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from dolfinx import fem

from vcell_fenics.backend.cell_force_balance import (
    ForceBalanceError,
    _is_zero,
    assemble_cell_force_balance,
    build_members_mesh,
    t9_equation,
)
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    MathDescription,
    MotionPrescribedVelocity,
    MotionUnknown,
    TemplateEquation,
)

MembraneKind = Literal["material", "front"]


@dataclass(frozen=True)
class MovingCellScope:
    """A model in stage 4a's scope: its member, the membrane bounding it, and how that membrane moves."""

    member: str
    membrane: str
    kind: MembraneKind
    tension: bool  # a material membrane with a tension: the force balance carries the semi-implicit term


def moving_membrane_of(md: MathDescription) -> str | None:
    """The membrane a T9 model moves, if any: one moving with the force balance's material (`motion: unknown`
    naming its velocity) or a growing front (a prescribed, non-zero velocity). None on a fixed skeleton."""

    from vcell_fenics.backend.geometry import prescribed_velocity

    eq = t9_equation(md)
    moving = [
        s.name
        for s in md.subdomains
        if s.kind == "surface"
        and (
            (isinstance(s.motion, MotionUnknown) and s.motion.variable == eq.variable and not _held(md, eq, s.name))
            or (isinstance(s.motion, MotionPrescribedVelocity) and prescribed_velocity(s) is not None)
        )
    ]
    return moving[0] if moving else None


def _held(md: MathDescription, eq: TemplateEquation, boundary: str) -> bool:
    return any(
        bc.variable == eq.variable
        and bc.boundary == boundary
        and isinstance(bc, BCDirichlet)
        and _is_zero(parse(bc.expression))
        for bc in md.boundary_conditions
    )


def moving_cell_scope(md: MathDescription, geometry: object, membrane: str) -> MovingCellScope:
    """Check that `md` on `geometry`, moving `membrane`, is in stage 4a's scope (ADR 023 decision 7), and return it.
    Refused by name: species or region variables (stage 5b), 3D, more than one process, several members, a member
    touching the box, any other membrane, and the membrane conditions with no use case yet."""

    from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry

    assert isinstance(geometry, MultiCompartmentGeometry)
    eq = t9_equation(md)

    def refuse(reason: str) -> ForceBalanceError:
        return ForceBalanceError(f"MovingCellScope: {reason} (stage 4a, ADR 023 decision 7)")

    others = [e for e in md.equations if e is not eq]
    if others:
        names = [e.variable for e in others if isinstance(e, TemplateEquation)]
        raise refuse(f"species or region variables {names} on a moving cell wait for stage 5b")
    if geometry.parent_mesh.geometry.dim != 2:
        raise refuse("a moving cell runs in 2D only; 3D is a later stage")
    if geometry.parent_mesh.comm.size > 1:
        raise refuse(
            f"a moving cell runs on one process: the slide, the contact check and the remesh are serial, and on "
            f"{geometry.parent_mesh.comm.size} processes the partition would skip the first two"
        )
    members = eq.members or (eq.subdomain,)
    if len(members) != 1:
        raise refuse(f"one member moves at a time; {list(members)} is stage 4b")
    (member,) = members
    if member not in geometry.compartments:
        raise refuse(f"the member {member!r} has no cells in the realized geometry {geometry.name!r}")
    if geometry.compartments[member].faces:
        raise refuse(f"the member {member!r} touches the box ({sorted(geometry.compartments[member].faces)}): stage 4c")
    if set(geometry.membranes) != {membrane}:
        raise refuse(
            f"the moving cell's membrane {membrane!r} must be the only one; {sorted(geometry.membranes)} is stage 4b"
        )
    part = geometry.membranes[membrane]
    if member not in (part.inside, part.outside):
        raise refuse(f"the moving membrane {membrane!r} does not bound the member {member!r}")
    sub = next(s for s in md.subdomains if s.name == membrane)
    on_membrane = [bc for bc in md.boundary_conditions if bc.variable == eq.variable and bc.boundary == membrane]
    surface = eq.surfaces.get(membrane, {})
    if isinstance(sub.motion, MotionUnknown):
        kind: MembraneKind = "material"
        if on_membrane:
            raise refuse(f"a velocity condition on the material membrane {membrane!r} has no use case yet")
    else:
        kind = "front"
        if "tension" in surface:
            raise refuse(f"the growing front {membrane!r} carries a tension; a front carries none (ADR 019 decision 5)")
        if "traction" in surface:
            raise refuse(f"an applied traction on the growing front {membrane!r} has no use case yet")
        if any(not (isinstance(bc, BCDirichlet) and _is_zero(parse(bc.expression))) for bc in on_membrane):
            raise refuse(f"a growing front {membrane!r} is free or held at {eq.variable} = 0, not given a velocity")
    return MovingCellScope(member, membrane, kind, kind == "material" and "tension" in surface)


@dataclass
class MovingCellResult:
    """Where a moving cell's run ended: the velocity solved there, the strides, the solves and, if it stopped for a
    remesh, the half-move still to do."""

    velocity: fem.Function
    final_time: float
    strides: int
    solves: int
    needs_remesh: bool = False
    pending_move: tuple[float, float] | None = None


def integrate_moving_cell(
    md: MathDescription,
    geometry: object,
    membrane: str,
    *,
    t_final: float,
    stride_ends: Sequence[float],
    output_times: Sequence[float] = (),
    on_output: Callable[[float, dict[str, fem.Function]], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    t_start: float = 0.0,
    pending_move: tuple[float, float] | None = None,
    stop_for_remesh: bool = False,
    velocity_override: Callable[[float, fem.Function], None] | None = None,
) -> MovingCellResult:
    """Move one cell by its force balance from ``t_start`` to ``t_final`` over ``stride_ends`` (the module
    docstring). ``on_output(t, {v: velocity})`` at each of ``output_times``, which must be stride ends, with the
    velocity solved on that geometry (it is the next stride's first solve). With ``stop_for_remesh`` it returns
    early, ``needs_remesh`` set, when a cell has degraded or a move would tangle the mesh (undone); a
    ``pending_move`` (the second half of a stride left undone) is then made first on the next geometry, by a
    forward-Euler half-move from a fresh solve, since the first half's velocity belongs to the old mesh."""

    from vcell_fenics.backend.discrete import MeshTangled
    from vcell_fenics.backend.moving_partition import FrontCannotAdvance, MovingPartition
    from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, _measures

    assert isinstance(geometry, MultiCompartmentGeometry)
    scope = moving_cell_scope(md, geometry, membrane)
    eq = t9_equation(md)
    ends = [float(e) for e in stride_ends]
    if not ends or abs(ends[-1] - t_final) > 1.0e-12 * max(1.0, abs(t_final)) or any(b <= a for a, b in pairwise(ends)):
        raise ValueError(f"stride_ends must rise strictly to t_final = {t_final}: {ends}")
    marks = [t_start, *ends]
    missed = [t for t in output_times if not any(abs(t - e) <= 1.0e-12 * max(1.0, abs(e)) for e in marks)]
    if on_output is not None and missed:
        raise ValueError(f"output times {missed} are not stride ends: a moving cell records only at a stride's end")

    balance = assemble_cell_force_balance(
        md, geometry, prebuilt=build_members_mesh(md, geometry), semi_implicit_dt=1.0 if scope.tension else None
    )
    velocity = balance.velocity
    motion = MovingPartition(
        md,
        geometry,
        membrane,
        traces={},
        vector_traces={eq.variable: velocity},
        velocity=f"trace({eq.variable})" if scope.kind == "material" else None,
    )
    size_constants = [*motion.size_constants, *balance.size_constants]
    surface_tags = {name: m.facet_tag for name, m in geometry.membranes.items()}

    def remeasure() -> None:
        geometry.sizes |= _measures(geometry, surface_tags)
        for name, constant in size_constants:
            constant.value = geometry.sizes[name]

    remeasure()  # the sizes of the geometry as handed over (a remeshed one included)
    solves = 0

    def solve(t_now: float, length: float) -> None:
        """`v` on the geometry as it is now, its tension linearized over the half-move it drives."""
        nonlocal solves
        if velocity_override is not None:
            velocity_override(t_now, velocity)
        else:
            balance.solve(t_now, semi_implicit_dt=length if scope.tension else None)
        solves += 1

    fresh = pending_move is not None or t_start > 0.0  # a geometry just remeshed: its first move must succeed

    def half_move(t_end: float, length: float, midpoint_of: np.ndarray | None = None) -> bool:
        """One half-move; False if the mesh needs a remesh first (the move undone)."""
        nonlocal fresh
        if stop_for_remesh and not fresh and motion.degraded():
            return False
        try:
            motion.advance(t_end, length, midpoint_of=midpoint_of)
        except MeshTangled as error:
            if fresh:
                raise FrontCannotAdvance(
                    f"the moving membrane {membrane!r} cannot advance at t = {t_end:g}: a freshly remeshed mesh "
                    f"tangled on its first move (vcell-fenics ADR 016 decision 6). {error}"
                ) from error
            if not stop_for_remesh:
                raise
            return False
        fresh = False
        remeasure()
        return True

    recorded = sorted(output_times) if on_output is not None else []

    def at_output(t_now: float) -> bool:
        return any(abs(t_now - r) <= 1.0e-12 * max(1.0, abs(r)) for r in recorded)

    def next_half(t_now: float) -> float:
        later = [e for e in ends if e > t_now + 1.0e-12 * max(1.0, abs(t_now))]
        return 0.5 * (later[0] - t_now) if later else 0.0

    t, strides, solved = t_start, 0, False
    if pending_move is not None:  # the second half of a stride, redone on the new mesh by forward Euler
        t_move, length = pending_move
        solve(t_move - length, length)
        if not half_move(t_move, length):
            return MovingCellResult(velocity, t, strides, solves, needs_remesh=True, pending_move=pending_move)
        t = t_move
    if on_output is not None and at_output(t):
        solve(t, next_half(t))
        solved = True
        on_output(t, {eq.variable: velocity})
    for t_end in ends:
        if t_end <= t + 1.0e-12 * max(1.0, abs(t)):
            continue
        half = 0.5 * (t_end - t)
        if not solved:
            solve(t, half)
        solved = False
        if not half_move(t + half, half):
            return MovingCellResult(velocity, t, strides, solves, needs_remesh=True)
        first = motion.membrane_velocity.x.array.copy()
        solve(t + half, half)
        if not half_move(t_end, half, midpoint_of=first):
            # the first half is done; the second waits for the new mesh (made there by forward Euler)
            return MovingCellResult(velocity, t + half, strides, solves, needs_remesh=True, pending_move=(t_end, half))
        t = t_end
        strides += 1
        if on_progress is not None:
            on_progress(t)
        if on_output is not None and at_output(t):
            solve(t, next_half(t))  # this stride's output is the next stride's first solve
            solved = True
            on_output(t, {eq.variable: velocity})
    if not solved:
        solve(t, 0.0)
    return MovingCellResult(velocity, t, strides, solves)


__all__ = [
    "MovingCellResult",
    "MovingCellScope",
    "integrate_moving_cell",
    "moving_cell_scope",
    "moving_membrane_of",
]
