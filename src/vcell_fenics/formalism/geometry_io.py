"""YAML / JSON ↔ `GeometryDescription` carriers — the geometry analogue of
`formalism.loader` / `formalism.dumper`.

Round-trip identity: ``load_geometry_dict(geometry_to_dict(g)) == g``. Defaults are omitted on
output (a `(0,0,0)` origin, empty subvolume/surface lists, a missing image) and re-applied on
load. The surface form is a ``geometry_description:`` envelope, mirroring the math
``math_description:`` form. The load path validates the schema dataclasses with a
``pydantic.TypeAdapter`` (boundary validation), so structural checks — required fields, unknown
fields, ``SubVolumeType`` membership, int/tuple coercion — are pydantic's, not hand-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    SubVolume,
)
from vcell_fenics.formalism.loader import FormalismLoadError

_DEFAULT_EXTENT = (1.0, 1.0, 1.0)
_DEFAULT_ORIGIN = (0.0, 0.0, 0.0)


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


# Validates the (plain stdlib) GeometryDescription dataclass tree at the boundary. Unknown-field
# rejection comes from each dataclass's `__pydantic_config__` (extra="forbid"); Literal / int /
# tuple coercion and required-field presence are pydantic's. Semantic checks (dim/type
# consistency, analytic-expression syntax) stay the validator's job, exactly as before.
_ADAPTER: TypeAdapter[GeometryDescription] = TypeAdapter(GeometryDescription)


def load_geometry_dict(raw: object) -> GeometryDescription:
    if not isinstance(raw, dict) or "geometry_description" not in raw:
        raise FormalismLoadError("", "missing top-level 'geometry_description:' key")
    try:
        return _ADAPTER.validate_python(raw["geometry_description"])
    except ValidationError as exc:
        raise _from_validation_error(exc) from exc


def load_geometry_yaml(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    return load_geometry_dict(yaml.safe_load(text))


def load_geometry_json(source: str | Path) -> GeometryDescription:
    text = Path(source).read_text() if isinstance(source, Path) else source
    return load_geometry_dict(json.loads(text))


def _from_validation_error(exc: ValidationError) -> FormalismLoadError:
    """Translate pydantic's first error into the package's path-bearing FormalismLoadError,
    rendering the location in the `geometry_description.subvolumes[0].type` style used elsewhere."""

    err = exc.errors()[0]
    path = "geometry_description"
    for part in err["loc"]:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return FormalismLoadError(path, err["msg"])
