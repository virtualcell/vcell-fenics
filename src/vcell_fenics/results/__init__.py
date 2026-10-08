"""Results bundles — the solver's output format ([ADR 010](../../../docs/decisions/010-results-bundle-vtu-zarr.md)).

A bundle is a zarr-v2 group directory: one VTK XML UnstructuredGrid per domain (written once for a fixed
mesh), each variable a ``(T, N)`` P1 point-data array whose columns follow that VTU's point order, per-time
statistics, and a manifest (``.zattrs["vcell_fenics"]``) that says which rows exist. Readable while the
run is still writing; ParaView gets it through :mod:`vcell_fenics.results.export`.
"""

from vcell_fenics.results.gather import P1Layout
from vcell_fenics.results.reader import Bundle
from vcell_fenics.results.recorder import BundleRecorder
from vcell_fenics.results.schema import (
    SCHEMA_VERSION,
    STATS_COLUMNS,
    Adjacency,
    BundleSchemaError,
    DomainInfo,
    Manifest,
    Segment,
    SolverInfo,
    SourceInfo,
    VariableInfo,
)
from vcell_fenics.results.writer import BundleWriter

__all__ = [
    "SCHEMA_VERSION",
    "STATS_COLUMNS",
    "Adjacency",
    "Bundle",
    "BundleRecorder",
    "BundleSchemaError",
    "BundleWriter",
    "DomainInfo",
    "Manifest",
    "P1Layout",
    "Segment",
    "SolverInfo",
    "SourceInfo",
    "VariableInfo",
]
