"""VCell ``MathOverrides``: per-job constant overrides and parameter scans, resolved exactly as VCell does.

A VCell simulation may override math constants — plainly (``<Constant Name="k">expr</Constant>``) or as
a **scan** over several values (``<Constant Name="k" ConstantArraySpec="1000|1001">…</Constant>``). A
scanned simulation runs as many jobs; each job's ``JobIndex`` selects one point of the scan. The
mapping is "immortalized by stored simulation job datasets" (VCell's own comment), so this is a port
of the Java, not a reinterpretation:

- ``scanIndex = jobIndex % scanCount`` (``Simulation.getScanIndex``), where ``scanCount`` is the
  product of every scanned constant's value count;
- the scanned names are **sorted**, and ``scanIndex`` is an odometer over them with the *first* name
  varying slowest (``MathOverrides.scanIndexToScanParameterCoordinate``);
- a list spec (type 1000) is its comma-separated values — old style ``1, 2, 3`` or new style
  ``"KMOLE", "KMOLE*2"``; an interval spec (type 1001) is ``min to max, [log, ]N values`` with value
  ``i`` equal to ``min + (max − min)·i/(N−1)``, or ``min·(max/min)^(i/(N−1))`` when logarithmic
  (``ConstantArraySpec.createFromString`` / ``createIntervalSpec``).

Sources: ``vcell-core/src/main/java/cbit/vcell/solver/{MathOverrides,ConstantArraySpec,Simulation}.java``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

LIST_SPEC = 1000
INTERVAL_SPEC = 1001


class OverrideError(ValueError):
    """A MathOverrides entry that cannot be parsed or applied."""


@dataclass(frozen=True)
class ScanSpec:
    """One scanned constant: its candidate values as VCell expression strings, in scan order."""

    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class MathOverrides:
    plain: Mapping[str, str]  # name → expression
    scans: tuple[ScanSpec, ...]

    @property
    def scan_count(self) -> int:
        return math.prod(len(scan.values) for scan in self.scans) if self.scans else 1


def parse_math_overrides(entries: Iterable[tuple[str, str | None, str]]) -> MathOverrides:
    """Build :class:`MathOverrides` from ``(name, ConstantArraySpec attribute or None, text)`` triples —
    the ``<Constant>`` children of a ``<MathOverrides>`` element."""

    plain: dict[str, str] = {}
    scans: list[ScanSpec] = []
    for name, spec_type, text in entries:
        body = text.strip()
        if spec_type is None:
            if not body:
                raise OverrideError(f"override of {name!r} has no expression")
            plain[name] = body
        elif int(spec_type) == LIST_SPEC:
            scans.append(ScanSpec(name, _list_values(name, body)))
        elif int(spec_type) == INTERVAL_SPEC:
            scans.append(ScanSpec(name, _interval_values(name, body)))
        else:
            raise OverrideError(f"override of {name!r} has unknown ConstantArraySpec type {spec_type!r}")
    return MathOverrides(plain=plain, scans=tuple(scans))


def scan_coordinates(scan_index: int, bounds: tuple[int, ...]) -> tuple[int, ...]:
    """``MathOverrides.scanIndexToScanParameterCoordinate``: ``bounds[i]`` is the highest zero-based index
    along axis ``i`` (a 5×6×7 scan has bounds ``(4, 5, 6)``); axis 0 varies slowest."""

    index = scan_index
    coordinates: list[int] = []
    for i in range(len(bounds)):
        offset = math.prod(b + 1 for b in bounds[i + 1 :])
        coordinate = index // offset
        if coordinate > bounds[i]:
            raise OverrideError(f"scan index {scan_index} is out of range for a {tuple(b + 1 for b in bounds)} scan")
        coordinates.append(coordinate)
        index -= offset * coordinate
    return tuple(coordinates)


def resolve_overrides(overrides: MathOverrides, job_index: int) -> dict[str, str]:
    """Every overridden constant's expression for job ``job_index`` — plain overrides as written, each
    scanned constant at this job's scan coordinate."""

    resolved = dict(overrides.plain)
    if overrides.scans:
        scans = sorted(overrides.scans, key=lambda scan: scan.name)  # "must do things in a consistent way"
        scan_index = job_index % overrides.scan_count
        coordinates = scan_coordinates(scan_index, tuple(len(scan.values) - 1 for scan in scans))
        for scan, coordinate in zip(scans, coordinates, strict=True):
            resolved[scan.name] = scan.values[coordinate]
    return resolved


def apply_overrides(math_description: Any, values: Mapping[str, str]) -> Any:
    """Return a copy of a pyvcell ``MathDescription`` whose ``constants`` carry the override expressions.
    Overriding a name that is not a constant of the math is an error — never a silent no-op."""

    if not values:
        return math_description
    known = {constant.name for constant in math_description.constants}
    unknown = sorted(set(values) - known)
    if unknown:
        raise OverrideError(f"MathOverrides name constants the math does not define: {unknown}")
    constants = [
        constant.model_copy(update={"exp": values[constant.name]}) if constant.name in values else constant
        for constant in math_description.constants
    ]
    return math_description.model_copy(update={"constants": constants})


def _list_values(name: str, body: str) -> tuple[str, ...]:
    if '"' in body:  # new style: "KMOLE", "KMOLE*2", "KMOLE*3"
        values = tuple(token.strip() for token in body.split('"') if token.strip() not in ("", ","))
    else:  # old style: 1, 2, 3
        values = tuple(token.strip() for token in body.split(",") if token.strip())
    if len(values) < 2:
        raise OverrideError(f"scan list for {name!r} needs at least two values, got {body!r}")
    return values


def _interval_values(name: str, body: str) -> tuple[str, ...]:
    to = body.find(" to ")
    if to < 0:
        raise OverrideError(f"invalid interval scan for {name!r}: {body!r}")
    low, rest = body[:to].strip(), body[to + 4 :]
    log_interval = "log," in rest
    rest = rest.replace("log,", "")
    comma = rest.rfind(",")
    if comma < 0:
        raise OverrideError(f"invalid interval scan for {name!r}: {body!r}")
    high = rest[:comma].strip()
    try:
        count = int(rest[comma + 1 :].split()[0])
    except (IndexError, ValueError):
        raise OverrideError(f"invalid value count in interval scan for {name!r}: {body!r}") from None
    if count < 1:
        raise OverrideError(f"interval scan for {name!r} needs at least one value")

    low_value, high_value = _number(low), _number(high)
    if log_interval and low_value is not None and high_value is not None and low_value * high_value <= 0:
        raise OverrideError(f"log interval for {name!r} needs a non-zero min and max of one sign")
    if count == 1 and not log_interval:
        return (low,)
    values: list[str] = []
    for i in range(count):
        fraction = i / (count - 1) if count > 1 else 0.0
        if low_value is not None and high_value is not None:  # evaluate as Java's Expression would
            value = (
                low_value * (high_value / low_value) ** fraction
                if log_interval
                else low_value + (high_value + -low_value) * fraction
            )
            values.append(repr(value))
        elif log_interval:
            values.append(f"(({low}) * pow((({high}) / ({low})), {fraction!r}))")
        else:
            values.append(f"(({low}) + ((({high}) - ({low})) * {fraction!r}))")
    return tuple(values)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


__all__ = [
    "INTERVAL_SPEC",
    "LIST_SPEC",
    "MathOverrides",
    "OverrideError",
    "ScanSpec",
    "apply_overrides",
    "parse_math_overrides",
    "resolve_overrides",
    "scan_coordinates",
]
