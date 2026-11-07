"""Translate a VCell geometry (`pyvcell.vcml.models_geometry.Geometry`) into a formalism
:class:`~vcell_fenics.formalism.geometry_schema.GeometryDescription` (ADR 007,
`docs/modeling/geometric-formalism.md` §2).

The geometry analogue of :func:`~vcell_fenics.pyvcell_bridge.importer.import_math_description`:
maps construct-for-construct over the (typed) pydantic object. Subvolumes carry over with their
type (analytic / csg / image / compartmental); analytic expressions are translated through the
same coordinate/time rules as the math importer (`x/y/z` → `geom.x[…]`); surface classes become
ordered membrane pairs; the image carries its metadata and its voxels (VCell's hex-zlib encoding,
unchanged). Nothing is rejected: every VCell geometry maps, even where its subvolume type cannot yet
be *meshed* (the realization layer's concern), so the data imports losslessly.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SubVolumeType,
    SurfaceClass,
)
from vcell_fenics.pyvcell_bridge.expression import translate_expression

if TYPE_CHECKING:
    from pyvcell.vcml.models_geometry import Geometry as VcmlGeometry
    from pyvcell.vcml.models_geometry import Image as VcmlImage
    from pyvcell.vcml.models_geometry import SubVolume as VcmlSubVolume
    from pyvcell.vcml.models_math import MathDescription as VcmlMathDescription


def import_geometry(vcml: VcmlGeometry, *, name: str | None = None) -> GeometryDescription:
    """Translate a pyvcell ``Geometry`` into a formalism ``GeometryDescription``. ``name``
    overrides the geometry's name (defaults to the VCell geometry's ``name``)."""

    return GeometryDescription(
        name=name if name is not None else vcml.name,
        dim=int(vcml.dim),
        extent=(float(vcml.extent[0]), float(vcml.extent[1]), float(vcml.extent[2])),
        origin=(float(vcml.origin[0]), float(vcml.origin[1]), float(vcml.origin[2])),
        subvolumes=tuple(_subvolume(sv) for sv in vcml.subvolumes),
        surfaces=tuple(
            SurfaceClass(name=sc.name, inside=sc.subvolume_ref_1, outside=sc.subvolume_ref_2)
            for sc in vcml.surface_classes
        ),
        image=_image(vcml.image) if vcml.image is not None else None,
    )


def complete_surfaces(gd: GeometryDescription, vcml_math: VcmlMathDescription) -> GeometryDescription:
    """``gd`` with a surface class for every membrane subdomain of ``vcml_math`` it does not declare.

    An older VCell geometry declares no ``<SurfaceClass>``: VCell derives each one from the geometry's
    membrane regions (``SurfaceClass.createName``), and its generated math names the derived class as a
    ``MembraneSubDomain`` with its ``inside_compartment`` / ``outside_compartment`` -- the two
    subvolumes, in the order that fixes the normal. pyvcell's geometry model carries no regions, so the
    math is where the derived classes are recorded; without them a membrane subdomain has no matching
    surface in the geometry and the model fails validation (§1.11.10). A membrane whose sides are not
    two subvolumes of ``gd`` is left alone (the validator reports it)."""

    have = {sc.name for sc in gd.surfaces}
    subvolumes = {sv.name for sv in gd.subvolumes}
    extra = tuple(
        SurfaceClass(name=m.name, inside=m.inside_compartment, outside=m.outside_compartment)
        for m in vcml_math.membrane_subdomains
        if m.name not in have
        and m.inside_compartment in subvolumes
        and m.outside_compartment in subvolumes
        and m.inside_compartment != m.outside_compartment
    )
    return replace(gd, surfaces=gd.surfaces + extra) if extra else gd


def _subvolume(sv: VcmlSubVolume) -> SubVolume:
    # `subvolume_type` is a `SubVolumeType` StrEnum; `.value` is its string form ("analytic", …).
    type_: SubVolumeType = sv.subvolume_type.value
    expression = translate_expression(sv.analytic_expr) if sv.analytic_expr else None
    return SubVolume(name=sv.name, type=type_, expression=expression, pixel_value=sv.image_pixel_value)


def _image(image: VcmlImage) -> GeometryImage:
    return GeometryImage(
        name=image.name,
        size=(int(image.size[0]), int(image.size[1]), int(image.size[2])),
        pixel_classes=tuple(PixelClass(name=p.name, pixel_value=int(p.pixel_value)) for p in image.pixel_classes),
        compressed_content=image.compressed_content or None,
    )
