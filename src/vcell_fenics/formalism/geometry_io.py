"""YAML / JSON ↔ `GeometryDescription` carriers — the geometry analogue of
`formalism.loader` / `formalism.dumper`.

Round-trip identity: ``load_geometry_dict(geometry_to_dict(g)) == g``. Defaults are omitted on
output (a `(0,0,0)` origin, empty subvolume/surface lists, a missing image) and re-applied on
load. The surface form is a ``geometry_description:`` envelope, mirroring the math
``math_description:`` form. Untyped parsed input is typed as ``object`` and narrowed at each
step (no ``Any``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml

from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SubVolumeType,
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


def geometry_to_dict(geometry: GeometryDescription) -> dict[str, object]:
    """A `GeometryDescription` as a plain dict under the `geometry_description:` envelope,
    with defaults omitted."""

    body: dict[str, object] = {"name": geometry.name, "dim": geometry.dim}
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


def _subvolume_to_dict(subvolume: SubVolume) -> dict[str, object]:
    out: dict[str, object] = {"name": subvolume.name, "type": subvolume.type}
    if subvolume.expression is not None:
        out["expression"] = subvolume.expression
    if subvolume.pixel_value is not None:
        out["pixel_value"] = subvolume.pixel_value
    return out


def _image_to_dict(image: GeometryImage) -> dict[str, object]:
    out: dict[str, object] = {"name": image.name, "size": list(image.size)}
    if image.pixel_classes:
        out["pixel_classes"] = [{"name": p.name, "pixel_value": p.pixel_value} for p in image.pixel_classes]
    return out


def dump_geometry_yaml(geometry: GeometryDescription) -> str:
    return yaml.safe_dump(geometry_to_dict(geometry), sort_keys=False, allow_unicode=True)


def dump_geometry_json(geometry: GeometryDescription) -> str:
    return json.dumps(geometry_to_dict(geometry), indent=2)


# -- load -----------------------------------------------------------------------


def load_geometry_dict(raw: dict[str, object]) -> GeometryDescription:
    if "geometry_description" not in raw:
        raise FormalismLoadError("", "missing top-level 'geometry_description:' key")
    body = _as_dict(raw["geometry_description"], "geometry_description")
    _reject_unknown("geometry_description", body, _GEOMETRY_FIELDS)
    return GeometryDescription(
        name=_str(body, "name", "geometry_description"),
        dim=_int(body, "dim", "geometry_description"),
        extent=_triple(body.get("extent", list(_DEFAULT_EXTENT)), "geometry_description.extent"),
        origin=_triple(body.get("origin", list(_DEFAULT_ORIGIN)), "geometry_description.origin"),
        subvolumes=tuple(
            _load_subvolume(s, f"geometry_description.subvolumes[{i}]")
            for i, s in enumerate(_dict_list(body.get("subvolumes", []), "geometry_description.subvolumes"))
        ),
        surfaces=tuple(
            _load_surface(s, f"geometry_description.surfaces[{i}]")
            for i, s in enumerate(_dict_list(body.get("surfaces", []), "geometry_description.surfaces"))
        ),
        image=_load_image(body["image"]) if body.get("image") is not None else None,
    )


def load_geometry_yaml(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    return load_geometry_dict(_as_dict(yaml.safe_load(text), ""))


def load_geometry_json(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    return load_geometry_dict(_as_dict(json.loads(text), ""))


def _load_subvolume(raw: dict[str, object], path: str) -> SubVolume:
    _reject_unknown(path, raw, _SUBVOLUME_FIELDS)
    return SubVolume(
        name=_str(raw, "name", path),
        type=cast("SubVolumeType", _str(raw, "type", path)),
        expression=_opt_str(raw.get("expression"), f"{path}.expression"),
        pixel_value=_opt_int(raw.get("pixel_value"), f"{path}.pixel_value"),
    )


def _load_surface(raw: dict[str, object], path: str) -> SurfaceClass:
    _reject_unknown(path, raw, _SURFACE_FIELDS)
    return SurfaceClass(
        name=_str(raw, "name", path),
        inside=_str(raw, "inside", path),
        outside=_str(raw, "outside", path),
    )


def _load_image(raw: object) -> GeometryImage:
    body = _as_dict(raw, "geometry_description.image")
    _reject_unknown("geometry_description.image", body, _IMAGE_FIELDS)
    return GeometryImage(
        name=_str(body, "name", "geometry_description.image"),
        size=_int_triple(body.get("size", [0, 0, 0]), "geometry_description.image.size"),
        pixel_classes=tuple(
            PixelClass(name=_str(p, "name", "pixel_classes"), pixel_value=_int(p, "pixel_value", "pixel_classes"))
            for p in _dict_list(body.get("pixel_classes", []), "geometry_description.image.pixel_classes")
        ),
    )


# -- typed accessors over untyped (`object`) parsed input -----------------------


def _as_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FormalismLoadError(path, f"expected a mapping, got {type(value).__name__}")
    return cast("dict[str, object]", value)


def _dict_list(value: object, path: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise FormalismLoadError(path, f"expected a list, got {type(value).__name__}")
    return [_as_dict(item, f"{path}[{i}]") for i, item in enumerate(value)]


def _str(d: dict[str, object], key: str, path: str) -> str:
    value = _require(d, key, path)
    if not isinstance(value, str):
        raise FormalismLoadError(path, f"field {key!r} must be a string, got {type(value).__name__}")
    return value


def _int(d: dict[str, object], key: str, path: str) -> int:
    value = _require(d, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalismLoadError(path, f"field {key!r} must be an integer, got {type(value).__name__}")
    return value


def _opt_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FormalismLoadError(path, f"expected a string, got {type(value).__name__}")
    return value


def _opt_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalismLoadError(path, f"expected an integer, got {type(value).__name__}")
    return value


def _require(d: dict[str, object], key: str, path: str) -> object:
    if key not in d:
        raise FormalismLoadError(path, f"missing required field {key!r}")
    return d[key]


def _reject_unknown(path: str, raw: dict[str, object], allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise FormalismLoadError(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _triple(value: object, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise FormalismLoadError(path, "expected a list of three numbers")
    return (_num(value[0], path), _num(value[1], path), _num(value[2], path))


def _int_triple(value: object, path: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise FormalismLoadError(path, "expected a list of three integers")
    return (_as_int(value[0], path), _as_int(value[1], path), _as_int(value[2], path))


def _num(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalismLoadError(path, f"expected a number, got {type(value).__name__}")
    return float(value)


def _as_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalismLoadError(path, f"expected an integer, got {type(value).__name__}")
    return value
