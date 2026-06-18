"""Realize a :class:`~vcell_fenics.formalism.geometry_schema.GeometryDescription` into a concrete
backend :class:`~vcell_fenics.backend.geometry.Geometry` (ADR 007; `geometric-formalism.md` §3).

This is the bridge from the declarative geometry *spec* to a meshed object the backend solves on —
the realization layer §3.1 describes. It subsumes the imperative ``make_*`` helpers as recipes over
the formalism.

**Realization v1, step 1 (this module): the trivial / non-spatial case.** A ``dim = 0`` geometry is
well-mixed: each ``compartmental`` subvolume is a lumped compartment with no spatial extent, and the
lumped-ODE templates need no real mesh. Each such subvolume (and any membrane between them) is backed
by a minimal single-cell mesh that carries its value as a constant — a representational stand-in, not
a spatial domain.

**Not yet here (step 2):** body-fitted meshing of ``csg`` / primitive-``analytic`` subvolumes via gmsh
OCC (and the level-set path for arbitrary analytic, on top of `formalism/rvachev.py`). A spatial
description raises :class:`NotImplementedError` until that slice lands.
"""

from __future__ import annotations

from dolfinx import mesh as dmesh
from mpi4py import MPI

from vcell_fenics.backend.geometry import Geometry, SubdomainGeometry
from vcell_fenics.formalism.geometry_schema import GeometryDescription


class RealizationError(ValueError):
    """A :class:`GeometryDescription` is structurally well-formed but cannot be realized into a
    backend :class:`Geometry` (e.g. a non-spatial geometry carrying a spatial subvolume type)."""


def realize(description: GeometryDescription, *, comm: MPI.Comm = MPI.COMM_WORLD) -> Geometry:
    """Realize ``description`` into a backend :class:`Geometry`.

    v1 implements the ``dim = 0`` trivial case; spatial descriptions raise
    :class:`NotImplementedError` (the body-fitted gmsh-OCC slice is next). The result is *not*
    registered — the caller passes it to
    :func:`~vcell_fenics.backend.geometry.register_geometry` if name resolution is wanted.
    """

    if description.dim == 0:
        return _realize_trivial(description, comm)
    raise NotImplementedError(
        f"realization of dim={description.dim} geometry {description.name!r} is not implemented yet "
        "(Realization v1 step 2: body-fitted gmsh-OCC meshing of csg / analytic subvolumes)"
    )


def _realize_trivial(description: GeometryDescription, comm: MPI.Comm) -> Geometry:
    if not description.subvolumes:
        raise RealizationError(f"geometry {description.name!r} has no subvolumes to realize")

    subdomains: dict[str, SubdomainGeometry] = {}
    # Every subvolume of a non-spatial geometry is a well-mixed compartment (a `volume` subdomain).
    for subvolume in description.subvolumes:
        if subvolume.type != "compartmental":
            raise RealizationError(
                f"non-spatial (dim 0) geometry {description.name!r} has subvolume {subvolume.name!r} "
                f"of type {subvolume.type!r}; a well-mixed geometry's subvolumes must be 'compartmental'"
            )
        subdomains[subvolume.name] = SubdomainGeometry(mesh=_lumped_mesh(comm), kind="volume")

    # A membrane between two well-mixed compartments is itself non-spatial (a `surface` subdomain).
    for surface in description.surfaces:
        subdomains[surface.name] = SubdomainGeometry(mesh=_lumped_mesh(comm), kind="surface")

    return Geometry(name=description.name, subdomains=subdomains)


def _lumped_mesh(comm: MPI.Comm) -> dmesh.Mesh:
    """A minimal single-cell mesh standing in for a non-spatial, well-mixed compartment: the
    lumped-ODE templates carry the compartment's value as a constant over this one cell."""

    return dmesh.create_unit_interval(comm, 1)
