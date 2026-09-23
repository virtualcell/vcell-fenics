"""Structural validation of a `GeometryDescription` — the geometry analogue of
`formalism.validator`.

Catches the construction-time errors a geometry can have independently of any mesh: a bad
dimension, a subvolume type that does not match the dimension, an analytic subvolume with a
malformed (or missing) expression, a surface referencing a subvolume that does not exist, name
collisions. Reuses the math validator's `Diagnostic` so both formalisms report findings the
same way. Realization concerns (can this analytic shape actually be meshed?) are not checked
here — that is the realization layer's job.
"""

from __future__ import annotations

import numpy as np

from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume
from vcell_fenics.formalism.parser import ExpressionSyntaxError, parse
from vcell_fenics.formalism.validator import Diagnostic, FormalismValidationError

_SUBVOLUME_TYPES = frozenset({"compartmental", "analytic", "csg", "image"})


def validate_geometry(geometry: GeometryDescription) -> list[Diagnostic]:
    """Return all structural findings (errors and warnings) for ``geometry``."""

    out: list[Diagnostic] = []

    if geometry.dim not in (0, 1, 2, 3):
        out.append(Diagnostic("error", "dim", f"dimension must be 0, 1, 2, or 3; got {geometry.dim}"))

    names: set[str] = set()
    for i, sub in enumerate(geometry.subvolumes):
        path = f"subvolumes[{i}]"
        if sub.name in names:
            out.append(Diagnostic("error", path, f"duplicate subvolume name {sub.name!r}"))
        names.add(sub.name)
        _check_subvolume(sub, path, geometry, out)

    surface_names: set[str] = set()
    for i, surf in enumerate(geometry.surfaces):
        path = f"surfaces[{i}]"
        if surf.name in surface_names:
            out.append(Diagnostic("error", path, f"duplicate surface name {surf.name!r}"))
        surface_names.add(surf.name)
        for role, ref in (("inside", surf.inside), ("outside", surf.outside)):
            if ref not in names:
                out.append(Diagnostic("error", path, f"{role} references unknown subvolume {ref!r}"))
        if surf.inside == surf.outside:
            out.append(Diagnostic("error", path, f"surface {surf.name!r} has the same subvolume on both sides"))

    # Dimension / type consistency.
    has_compartmental = any(s.type == "compartmental" for s in geometry.subvolumes)
    if geometry.dim == 0 and not has_compartmental and geometry.subvolumes:
        out.append(
            Diagnostic("error", "subvolumes", "a non-spatial (dim 0) geometry must use compartmental subvolumes")
        )
    if geometry.dim > 0 and has_compartmental:
        out.append(
            Diagnostic(
                "error", "subvolumes", "compartmental subvolumes are only valid in a non-spatial (dim 0) geometry"
            )
        )
    if geometry.dim == 0 and geometry.surfaces:
        out.append(Diagnostic("error", "surfaces", "a non-spatial (dim 0) geometry cannot have surfaces"))

    _check_image_voxels(geometry, out)
    return out


def _check_image_voxels(geometry: GeometryDescription, out: list[Diagnostic]) -> None:
    """When the image carries voxels: they decode to its size, every pixel value belongs to an image
    subvolume (VCell maps each value to a subvolume — an unmapped one would leave space unowned), and
    each image subvolume's value occurs (an empty subvolume has nothing to mesh)."""

    image = geometry.image
    if image is None or image.compressed_content is None:
        return
    try:
        voxels = image.voxels()
    except ValueError as exc:
        out.append(Diagnostic("error", "image.compressed_content", str(exc)))
        return
    present = {int(v) for v in np.unique(voxels)}
    mapped = {s.pixel_value for s in geometry.subvolumes if s.type == "image" and s.pixel_value is not None}
    for value in sorted(present - mapped):
        out.append(Diagnostic("error", "image", f"pixel value {value} occurs in the image but no subvolume maps it"))
    for i, sub in enumerate(geometry.subvolumes):
        if sub.type == "image" and sub.pixel_value is not None and sub.pixel_value not in present:
            out.append(Diagnostic("warning", f"subvolumes[{i}]", f"pixel_value {sub.pixel_value} occurs in no voxel"))


def _check_subvolume(sub: SubVolume, path: str, geometry: GeometryDescription, out: list[Diagnostic]) -> None:
    if sub.type not in _SUBVOLUME_TYPES:
        out.append(Diagnostic("error", path, f"unknown subvolume type {sub.type!r}"))
        return
    if sub.type == "analytic":
        if sub.expression is None:
            out.append(Diagnostic("error", path, "an analytic subvolume must have an 'expression'"))
        else:
            try:
                parse(sub.expression)
            except ExpressionSyntaxError as exc:
                out.append(Diagnostic("error", f"{path}.expression", f"malformed analytic expression: {exc}"))
    if sub.type == "image":
        if sub.pixel_value is None:
            out.append(Diagnostic("error", path, "an image subvolume must have a 'pixel_value'"))
        elif geometry.image is None:
            out.append(Diagnostic("error", path, "an image subvolume requires the geometry to carry an 'image'"))
        elif geometry.image.pixel_classes and all(
            pc.pixel_value != sub.pixel_value for pc in geometry.image.pixel_classes
        ):
            out.append(Diagnostic("warning", path, f"pixel_value {sub.pixel_value} matches no class in the image"))


def validate_geometry_or_raise(geometry: GeometryDescription) -> None:
    """Validate and raise `FormalismValidationError` if there is any error-severity finding."""

    errors = [d for d in validate_geometry(geometry) if d.severity == "error"]
    if errors:
        raise FormalismValidationError(errors)
