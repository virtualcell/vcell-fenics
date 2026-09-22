"""The results-bundle manifest, schema 1 ([ADR 010](../../../docs/decisions/010-results-bundle-vtu-zarr.md) §3).

A bundle is a zarr-v2 group directory whose root ``.zattrs`` holds ``{"vcell_fenics": <Manifest>}``.
The manifest is the bundle's self-description — domains, variables, the output times written so far,
run status — and the one thing a reader must consult before touching an array: ``times`` is
authoritative (arrays are preallocated to ``planned_times``, and an unwritten row reads as NaN).

The classes are plain frozen dataclasses (the repo's schema idiom, as in ``formalism/``), validated at
the JSON boundary by a pydantic ``TypeAdapter``. Unlike the formalism they **ignore** unknown keys, so
an older reader still opens a bundle a newer writer annotated further (reader rules, ADR 010 §3). A
newer ``schema`` number is refused outright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter

SCHEMA_VERSION = 1
MANIFEST_KEY = "vcell_fenics"
STATS_COLUMNS = ("mean", "total", "min", "max")

_IGNORE_EXTRA = ConfigDict(extra="ignore")

DomainKind: TypeAlias = Literal["volume", "membrane"]
Status: TypeAlias = Literal["running", "completed", "failed"]
Profile: TypeAlias = Literal["fixed", "segmented"]
Motion: TypeAlias = Literal["none", "ale"]


@dataclass(frozen=True, slots=True)
class DomainInfo:
    """One VCell domain (a CompartmentSubDomain or MembraneSubDomain) and its mesh file."""

    __pydantic_config__ = _IGNORE_EXTRA
    kind: DomainKind
    dim: int  # topological dimension of the domain's cells
    gdim: int  # geometric dimension of the points (the VTU always stores 3 components)
    mesh: str  # bundle-relative path of the domain's VTU (per segment, under its prefix)
    n_points: int
    n_cells: int
    cell_type: int  # VTK cell type: 3 line, 5 triangle, 10 tetra


@dataclass(frozen=True, slots=True)
class VariableInfo:
    """A field on one domain: P1 point data whose columns follow the domain VTU's point order."""

    __pydantic_config__ = _IGNORE_EXTRA
    name: str
    domain: str
    path: str  # bundle-relative zarr array, (T, n_points)
    stats: str  # bundle-relative zarr array, (T, len(STATS_COLUMNS))
    assoc: Literal["point"] = "point"
    element: str = "P1"


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of output rows sharing one mesh topology. A fixed-mesh run is exactly one segment with an
    empty prefix; a remesh starts a new segment whose meshes and arrays live under ``prefix``."""

    __pydantic_config__ = _IGNORE_EXTRA
    index: int
    t0: float
    count: int
    motion: Motion = "none"
    prefix: str = ""


@dataclass(frozen=True, slots=True)
class SolverInfo:
    __pydantic_config__ = _IGNORE_EXTRA
    version: str
    dolfinx: str
    mpi_ranks: int
    options: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Where the model came from — for a VCell run, the SimulationTask identity."""

    __pydantic_config__ = _IGNORE_EXTRA
    kind: str  # "simtask" | "vcml" | "vcell-yaml" | "native"
    file: str | None = None
    sim_key: str | None = None
    job_index: int | None = None
    task_id: int | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    __pydantic_config__ = _IGNORE_EXTRA
    schema: int
    status: Status
    times: tuple[float, ...]
    planned_times: tuple[float, ...]
    segments: tuple[Segment, ...]
    domains: dict[str, DomainInfo]
    variables: tuple[VariableInfo, ...]
    solver: SolverInfo
    source: SourceInfo
    updated: str
    profile: Profile = "fixed"
    progress: float = 0.0
    message: str | None = None
    stats_columns: tuple[str, ...] = STATS_COLUMNS

    def variable(self, domain: str, name: str) -> VariableInfo:
        for variable in self.variables:
            if variable.domain == domain and variable.name == name:
                return variable
        raise KeyError(f"no variable {name!r} on domain {domain!r}")


_ADAPTER: TypeAdapter[Manifest] = TypeAdapter(Manifest)


class BundleSchemaError(ValueError):
    """The directory is not a results bundle this reader understands."""


def manifest_from_attrs(attrs: object) -> Manifest:
    """Validate a root ``.zattrs`` document into a :class:`Manifest`, refusing a newer schema."""

    if not isinstance(attrs, dict) or MANIFEST_KEY not in attrs:
        raise BundleSchemaError(f"no {MANIFEST_KEY!r} manifest in the bundle's root attributes")
    raw = attrs[MANIFEST_KEY]
    version = raw.get("schema") if isinstance(raw, dict) else None
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise BundleSchemaError(f"bundle schema {version!r} is not supported (this reader knows ≤ {SCHEMA_VERSION})")
    return _ADAPTER.validate_python(raw)


def manifest_to_attrs(manifest: Manifest) -> dict[str, Any]:
    return {MANIFEST_KEY: _ADAPTER.dump_python(manifest, mode="json")}


def manifest_json_schema() -> dict[str, Any]:
    """The manifest's JSON Schema — published as ``docs/results-bundle.schema.json`` for Java/JS readers."""

    schema: dict[str, Any] = _ADAPTER.json_schema()
    schema["$id"] = "https://github.com/virtualcell/vcell-fenics/docs/results-bundle.schema.json"
    schema["title"] = "vcell-fenics results bundle manifest (schema 1)"
    return schema


def dumps_json_schema() -> str:
    return json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n"
