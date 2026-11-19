"""Error tolerances that mean the same thing for every species (ADR 022).

PETSc's adaptive TS accepts a step when the weighted RMS of its local error estimate is at most 1, each entry
weighted by ``atolᵢ + rtolᵢ·max(|uᵢ|, |yᵢ|)``; an entry whose two tolerances are both zero is left out of the sum
and of the count (``VecErrorWeightedNorm2``, PETSc 3.25.1). One scalar ``atol`` across species of very different
sizes lets a small species' error pass unchecked, and the 2-norm over all unknowns lets a species with few unknowns
(a region variable) lose control as the mesh is refined. Here, per decision:

1. **Each species' absolute tolerance from its own scale:** ``atol_s = min(atol, rtol·scale_s)``, with ``scale_s``
   the largest ``|u|`` of species ``s`` so far over the whole domain (its initial values to start with, raised
   after every accepted step). A scale no larger than ``floor`` times the largest species' scale keeps ``atol``:
   by default ``floor`` is the machine epsilon, below which the species cannot be resolved beside the larger one in
   a shared solve and must not collapse the step. ``floor`` is a solver option (``--tolerance-floor``): raising it
   (say to 1e-6) stops a trace species, many decades below the largest, from setting the step size.
   The scales live in a `SpeciesScales` the integrator's caller owns, so they persist across strides, the
   persistent integrator and a restart after a remesh.
2. **Balanced vector tolerances:** species ``s``'s entries are multiplied by ``√(n_s/n)`` (``n_s`` its counted
   unknowns, ``n`` all counted unknowns, both global), so the 2-norm becomes ``√(Σ_s RMS_s²)``, every species
   weighed equally whatever its number of unknowns.
3. Unknowns held by a strong Dirichlet condition get ``atol = rtol = 0``: PETSc then leaves them out.

``scaling="scalar"`` keeps the scalar tolerances as given (the behaviour before ADR 022).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from petsc4py import PETSc

ToleranceScaling = Literal["species", "scalar"]
TOLERANCE_SCALINGS: tuple[str, ...] = get_args(ToleranceScaling)

# Decision 1's default floor: a species no larger than the round-off of the largest one keeps the scalar atol.
DEFAULT_TOLERANCE_FLOOR = float(np.finfo(np.float64).eps)


@dataclass
class SpeciesScales:
    """Each species' running maximum ``|u|`` over the whole domain, by name (ADR 022 decision 1 (b)). Owned by the
    integrator's caller so it outlives one integrator: it never falls. ``floor`` is the relative scale (to the largest
    species') at or below which a species keeps the scalar ``atol``."""

    peak: dict[str, float] = field(default_factory=dict)
    floor: float = DEFAULT_TOLERANCE_FLOOR

    def __post_init__(self) -> None:
        if not (np.isfinite(self.floor) and 0.0 <= self.floor < 1.0):
            raise ValueError(f"tolerance floor must lie in [0, 1), got {self.floor!r}")

    def raise_to(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if np.isfinite(value) and value > self.peak.get(name, 0.0):
                self.peak[name] = float(value)

    def species_atol(self, rtol: float, atol: float) -> dict[str, float]:
        """Each species' absolute tolerance: ``min(atol, rtol·scale)``, or ``atol`` for a scale at or below the
        floor (zero included)."""
        top = max(self.peak.values(), default=0.0)
        return {
            name: (min(atol, rtol * scale) if scale > self.floor * top and scale > 0.0 else atol)
            for name, scale in self.peak.items()
        }


def blocked_layout(
    blocks: Sequence[tuple[int, int, int, Sequence[str]]],
) -> dict[str, npt.NDArray[np.int64]]:
    """Each species' local indices in a blocked state vector. A block is ``(offset, size, bs, names)``: its owned
    entries start at ``offset``, number ``size`` and hold ``bs`` components interleaved by node, the ``k``-th named
    ``names[k]``. A component with no name (a placeholder block) is claimed by no species and stays out of the
    norm."""
    layout: dict[str, npt.NDArray[np.int64]] = {}
    for offset, size, bs, names in blocks:
        nodes = np.arange(size // bs, dtype=np.int64)
        for k, name in enumerate(names[:bs]):
            layout[name] = offset + k + bs * nodes
    return layout


class ToleranceControl:
    """The tolerances of one integrator's state vector, kept current with the species' scales.

    ``layout`` gives each species the local indices (into ``x``'s owned array) of its unknowns; every owned entry
    must belong to exactly one species. ``held`` are local indices of unknowns held by a strong Dirichlet condition.
    Build it once per state vector, `attach` it to every TS that integrates that vector, and `destroy` it with the
    vector."""

    def __init__(
        self,
        x: PETSc.Vec,
        layout: Mapping[str, npt.NDArray[np.integer[Any]]],
        *,
        rtol: float,
        atol: float,
        scaling: ToleranceScaling = "species",
        scales: SpeciesScales | None = None,
        held: npt.NDArray[np.integer[Any]] | None = None,
    ) -> None:
        if scaling not in TOLERANCE_SCALINGS:
            raise ValueError(f"tolerance scaling must be one of {TOLERANCE_SCALINGS}, got {scaling!r}")
        self.rtol, self.atol, self.scaling = rtol, atol, scaling
        self.scales = scales if scales is not None else SpeciesScales()
        self._comm: MPI.Comm = x.getComm().tompi4py()
        self._layout = {name: np.asarray(idx, dtype=np.int64) for name, idx in layout.items()}
        n_local = x.getLocalSize()
        counted = np.ones(n_local, dtype=bool)
        if held is not None and len(held):
            counted[np.asarray(held, dtype=np.int64)] = False
        self._counted = {name: idx[counted[idx]] for name, idx in self._layout.items()}
        local_counts = np.array([len(idx) for idx in self._counted.values()], dtype=np.int64)
        global_counts = np.zeros_like(local_counts)
        self._comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        self.counts = dict(zip(self._layout, (int(c) for c in global_counts), strict=True))
        total = sum(self.counts.values())
        self.weights = {name: (np.sqrt(n / total) if total else 1.0) for name, n in self.counts.items()}
        self._vatol: PETSc.Vec | None = None
        self._vrtol: PETSc.Vec | None = None
        if scaling == "species":
            self._vatol, self._vrtol = x.duplicate(), x.duplicate()
            self._vatol.set(0.0)
            self._vrtol.set(0.0)  # an unknown no species claims, or a held one, stays out of the norm
            for name, idx in self._counted.items():
                self._vrtol.array[idx] = rtol * self.weights[name]
        self.observe(x)

    def observe(self, x: PETSc.Vec) -> None:
        """Raise each species' scale to its present maximum (collectively) and refresh the absolute tolerances."""
        array = x.array_r
        local = np.array(
            [float(np.max(np.abs(array[idx]))) if len(idx) else 0.0 for idx in self._layout.values()],
            dtype=np.float64,
        )
        peak = np.zeros_like(local)
        self._comm.Allreduce(local, peak, op=MPI.MAX)
        self.scales.raise_to(dict(zip(self._layout, (float(p) for p in peak), strict=True)))
        if self._vatol is not None:
            atols = self.scales.species_atol(self.rtol, self.atol)
            for name, idx in self._counted.items():
                self._vatol.array[idx] = atols.get(name, self.atol) * self.weights[name]

    def attach(self, ts: PETSc.TS) -> None:
        """Give ``ts`` these tolerances; with species scaling, a post-step that keeps the scales current."""
        if self._vatol is None:
            ts.setTolerances(rtol=self.rtol, atol=self.atol)
            return
        # petsc4py accepts a Vec for either (TSSetTolerances' vrtol/vatol); its stub says float only
        ts.setTolerances(rtol=self._vrtol, atol=self._vatol)  # type: ignore[arg-type]
        ts.setPostStep(lambda ts_: self.observe(ts_.getSolution()))

    def species_atol(self) -> dict[str, float]:
        """Each species' absolute tolerance as decision 1 sets it (before balancing), or ``atol`` when scalar."""
        if self._vatol is None:
            return dict.fromkeys(self._layout, self.atol)
        atols = self.scales.species_atol(self.rtol, self.atol)
        return {name: atols.get(name, self.atol) for name in self._layout}

    def destroy(self) -> None:
        for vec in (self._vatol, self._vrtol):
            if vec is not None:
                vec.destroy()
        self._vatol = self._vrtol = None


def tolerance_summary(
    rtol: float, atol: float, scaling: ToleranceScaling, scales: SpeciesScales | None
) -> dict[str, Any]:
    """What a run's summary records about its error control (ADR 022 decision 4)."""
    out: dict[str, Any] = {"rtol": rtol, "atol": atol, "tolerance_scaling": scaling}
    if scaling == "species":
        out["error_norm"] = "2-norm, balanced per species"
        if scales is not None:
            atols = scales.species_atol(rtol, atol)
            out["tolerance_floor"] = scales.floor
            out["species_tolerances"] = {
                name: {"scale": scales.peak[name], "atol": atols[name]} for name in sorted(scales.peak)
            }
    else:
        out["error_norm"] = "2-norm over all unknowns"
    return out
