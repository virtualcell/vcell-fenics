"""Translate a VCell geometry (`pyvcell.vcml.models_geometry.Geometry`) into a formalism
:class:`~vcell_fenics.formalism.geometry_schema.GeometryDescription` (ADR 007,
`docs/modeling/geometric-formalism.md` §2).

The geometry analogue of :func:`~vcell_fenics.pyvcell_bridge.importer.import_math_description`:
duck-typed over the pydantic object, mapping construct-for-construct. Subvolumes carry over with
their type (analytic / csg / image / compartmental); analytic expressions are translated through
the same coordinate/time rules as the math importer (`x/y/z` → `geom.x[…]`); surface classes
become ordered membrane pairs; the image carries its metadata only (not the raw voxel blob — see
the schema). Nothing is rejected: every VCell geometry maps, even where its subvolume type cannot
yet be *meshed* (the realization layer's concern), so the data imports losslessly.
"""

from __future__ import annotations

from typing import Any

from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SubVolumeType,
    SurfaceClass,
)
from vcell_fenics.pyvcell_bridge.expression import translate_expression


def import_geometry(vcml: Any, *, name: str | None = None) -> GeometryDescription:
    """Translate a pyvcell ``Geometry`` into a formalism ``GeometryDescription``. ``name``
    overrides the geometry's name (defaults to the VCell geometry's ``name``)."""

    return GeometryDescription(
        name=name if name is not None else vcml.name,
        dim=int(vcml.dim),
        extent=_triple(vcml.extent),
        origin=_triple(vcml.origin),
        subvolumes=tuple(_subvolume(sv) for sv in vcml.subvolumes),
        surfaces=tuple(
            SurfaceClass(name=sc.name, inside=sc.subvolume_ref_1, outside=sc.subvolume_ref_2)
            for sc in vcml.surface_classes
        ),
        image=_image(vcml.image) if getattr(vcml, "image", None) is not None else None,
    )


def _subvolume(sv: Any) -> SubVolume:
    # `subvolume_type` is a `SubVolumeType` StrEnum; `.value` is its string form ("analytic", …).
    type_: SubVolumeType = sv.subvolume_type.value
    expression = translate_expression(sv.analytic_expr) if sv.analytic_expr else None
    return SubVolume(name=sv.name, type=type_, expression=expression, pixel_value=sv.image_pixel_value)


def _image(image: Any) -> GeometryImage:
    return GeometryImage(
        name=image.name,
        size=_int_triple(image.size),
        pixel_classes=tuple(PixelClass(name=p.name, pixel_value=int(p.pixel_value)) for p in image.pixel_classes),
    )


def _triple(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _int_triple(value: Any) -> tuple[int, int, int]:
    return (int(value[0]), int(value[1]), int(value[2]))
