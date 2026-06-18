"""YAML / JSON ↔ `GeometryDescription` carriers — the geometry analogue of
`formalism.loader` / `formalism.dumper`.

Round-trip identity: ``load_geometry_dict(geometry_to_dict(g)) == g``. Defaults are omitted on
output (a `(0,0,0)` origin, empty subvolume/surface lists, a missing image) and re-applied on
load. The surface form is a ``geometry_description:`` envelope, mirroring the math
``math_description:`` form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SurfaceClass,
)
from vcell_fenics.formalism.loader import FormalismLoadError

_DEFAULT_EXTENT = (1.0, 1.0, 1.0)
_DEFAULT_ORIGIN = (0.0, 0.0, 0.0)
_SUBVOLUME_FIELDS = {"name", "type", "expression", "pixel_value"}
_SURFACE_FIELDS = {"name", "inside", "outside"}
_IMAGE_FIELDS = {"name", "size", "pixel_classes"}
_GEOMETRY_FIELDS = {"name", "dim", "extent", "origin", "subvolumes", "surfaces", "image"}


# -- dump -----------------------------------------------------------------------


def geometry_to_dict(geometry: GeometryDescription) -> dict[str, Any]:
    """A `GeometryDescription` as a plain dict under the `geometry_description:` envelope,
    with defaults omitted."""

    body: dict[str, Any] = {"name": geometry.name, "dim": geometry.dim}
    if tuple(geometry.extent) != _DEFAULT_EXTENT:
        body["extent"] = list(geometry.extent)
    if tuple(geometry.origin) != _DEFAULT_ORIGIN:
        body["origin"] = list(geometry.origin)
    if geometry.subvolumes:
        body["subvolumes"] = [_subvolume_to_dict(s) for s in geometry.subvolumes]
    if geometry.surfaces:
        body["surfaces"] = [{"name": s.name, "inside": s.inside, "outside": s.outside} for s in geometry.surfaces]
    if geometry.image is not None:
        body["image"] = _image_to_dict(geometry.image)
    return {"geometry_description": body}


def _subvolume_to_dict(subvolume: SubVolume) -> dict[str, Any]:
    out: dict[str, Any] = {"name": subvolume.name, "type": subvolume.type}
    if subvolume.expression is not None:
        out["expression"] = subvolume.expression
    if subvolume.pixel_value is not None:
        out["pixel_value"] = subvolume.pixel_value
    return out


def _image_to_dict(image: GeometryImage) -> dict[str, Any]:
    out: dict[str, Any] = {"name": image.name, "size": list(image.size)}
    if image.pixel_classes:
        out["pixel_classes"] = [{"name": p.name, "pixel_value": p.pixel_value} for p in image.pixel_classes]
    return out


def dump_geometry_yaml(geometry: GeometryDescription) -> str:
    return yaml.safe_dump(geometry_to_dict(geometry), sort_keys=False, allow_unicode=True)


def dump_geometry_json(geometry: GeometryDescription) -> str:
    return json.dumps(geometry_to_dict(geometry), indent=2)


# -- load -----------------------------------------------------------------------


def load_geometry_dict(raw: dict[str, Any]) -> GeometryDescription:
    if "geometry_description" not in raw:
        raise FormalismLoadError("", "missing top-level 'geometry_description:' key")
    body = raw["geometry_description"]
    if not isinstance(body, dict):
        raise FormalismLoadError("geometry_description", f"expected a mapping, got {type(body).__name__}")
    _reject_unknown("geometry_description", body, _GEOMETRY_FIELDS)
    name = _require(body, "name", "geometry_description")
    dim = _require(body, "dim", "geometry_description")
    return GeometryDescription(
        name=name,
        dim=dim,
        extent=_triple(body.get("extent", list(_DEFAULT_EXTENT)), "geometry_description.extent"),
        origin=_triple(body.get("origin", list(_DEFAULT_ORIGIN)), "geometry_description.origin"),
        subvolumes=tuple(
            _load_subvolume(s, f"geometry_description.subvolumes[{i}]")
            for i, s in enumerate(body.get("subvolumes", []))
        ),
        surfaces=tuple(
            _load_surface(s, f"geometry_description.surfaces[{i}]") for i, s in enumerate(body.get("surfaces", []))
        ),
        image=_load_image(body["image"]) if body.get("image") is not None else None,
    )


def load_geometry_yaml(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise FormalismLoadError("", f"expected a mapping at the top level, got {type(raw).__name__}")
    return load_geometry_dict(raw)


def load_geometry_json(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise FormalismLoadError("", f"expected a mapping at the top level, got {type(raw).__name__}")
    return load_geometry_dict(raw)


def _load_subvolume(raw: Any, path: str) -> SubVolume:
    if not isinstance(raw, dict):
        raise FormalismLoadError(path, f"expected a mapping, got {type(raw).__name__}")
    _reject_unknown(path, raw, _SUBVOLUME_FIELDS)
    return SubVolume(
        name=_require(raw, "name", path),
        type=_require(raw, "type", path),
        expression=raw.get("expression"),
        pixel_value=raw.get("pixel_value"),
    )


def _load_surface(raw: Any, path: str) -> SurfaceClass:
    if not isinstance(raw, dict):
        raise FormalismLoadError(path, f"expected a mapping, got {type(raw).__name__}")
    _reject_unknown(path, raw, _SURFACE_FIELDS)
    return SurfaceClass(
        name=_require(raw, "name", path),
        inside=_require(raw, "inside", path),
        outside=_require(raw, "outside", path),
    )


def _load_image(raw: Any) -> GeometryImage:
    if not isinstance(raw, dict):
        raise FormalismLoadError("geometry_description.image", f"expected a mapping, got {type(raw).__name__}")
    _reject_unknown("geometry_description.image", raw, _IMAGE_FIELDS)
    size = raw.get("size", [0, 0, 0])
    if not isinstance(size, list) or len(size) != 3:
        raise FormalismLoadError("geometry_description.image.size", "expected a list of three integers")
    return GeometryImage(
        name=_require(raw, "name", "geometry_description.image"),
        size=(int(size[0]), int(size[1]), int(size[2])),
        pixel_classes=tuple(
            PixelClass(
                name=_require(p, "name", "pixel_classes"), pixel_value=int(_require(p, "pixel_value", "pixel_classes"))
            )
            for p in raw.get("pixel_classes", [])
        ),
    )


def _require(raw: dict[str, Any], key: str, path: str) -> Any:
    if key not in raw:
        raise FormalismLoadError(path, f"missing required field {key!r}")
    return raw[key]


def _reject_unknown(path: str, raw: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise FormalismLoadError(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _triple(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise FormalismLoadError(path, "expected a list of three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))
