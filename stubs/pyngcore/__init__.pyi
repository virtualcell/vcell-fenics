"""Minimal local stub for the pyngcore subset vcell-fenics uses (pyngcore ships no
static types). Scoped to our call sites, not a full stub of the package."""

def SetNumThreads(threads: int) -> None:
    """Cap Netgen's TaskManager worker pool. We call ``SetNumThreads(1)`` before the
    mesher loads: our meshes are small/serial and the default pool busy-waits in a
    long-lived process (ADR 008 §3)."""
    ...
