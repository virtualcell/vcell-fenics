"""YAML / JSON → MathDescription loader.

Parses the surface form documented in §2.1 / §2.2 of
docs/modeling/declarative-formalism.md into the dataclass schema. The
loader handles structural concerns only:

- Top-level `math_description:` envelope unwrapping.
- Boundary validation of the schema dataclass tree via a `pydantic.TypeAdapter`.
  The discriminated unions (Motion, Parameter, Equation, BoundaryCondition) and
  their dispatch rules — `kind` tags, `expression`-vs-`value` field presence,
  unknown-field rejection, numeric→string coercion — live on the schema classes
  (`formalism.schema`); this module just drives the adapter and maps its
  `ValidationError`s back to `FormalismLoadError`.

Semantic validation (name resolution, type checking, temporality
consistency, etc.) belongs to the validator module and is intentionally
not done here. The loader's contract is "structurally well-formed YAML/JSON
→ structurally well-formed dataclass tree."

Errors are raised as `FormalismLoadError` carrying a JSON-pointer-style
path string indicating where in the document the problem occurred, so
debugging large models is tractable.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from vcell_fenics.formalism.schema import MathDescription


class FormalismLoadError(Exception):
    """A MathDescription document is structurally malformed.

    `path` is a JSON-pointer-style string indicating the offending field
    in the source document (e.g. `math_description.parameters[3].kind`).
    `message` is the human-readable explanation. The combined str()
    presentation reads "<path>: <message>".
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


# Validates the MathDescription dataclass tree at the YAML/JSON boundary. All the structural
# rules (required/unknown fields, union dispatch, coercion) are declared on the schema classes.
_ADAPTER: TypeAdapter[MathDescription] = TypeAdapter(MathDescription)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def load_yaml(source: str | Path) -> MathDescription:
    """Parse a YAML document or file into a MathDescription.

    `source` is either YAML text (str) or a file path. A heuristic
    distinguishes them: anything that contains a newline or has no
    plausible filesystem path is treated as text; everything else is
    opened as a file. To force the file-path interpretation, wrap the
    string in `pathlib.Path`.
    """

    return load_dict(yaml.safe_load(_read_text(source)))


def load_json(source: str | Path) -> MathDescription:
    """Parse a JSON document or file into a MathDescription.

    Same source-interpretation heuristic as :func:`load_yaml`.
    """

    return load_dict(json.loads(_read_text(source)))


def load_dict(raw: object) -> MathDescription:
    """Parse an already-deserialised dict (e.g. from YAML/JSON) into a
    MathDescription. The dict must have the top-level
    `math_description:` envelope.
    """

    if not isinstance(raw, dict) or "math_description" not in raw:
        raise FormalismLoadError("", "missing top-level 'math_description:' envelope (the doc's §2.1.2 schema)")
    try:
        return _ADAPTER.validate_python(raw["math_description"])
    except ValidationError as exc:
        raise _from_validation_error(exc) from exc


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _from_validation_error(exc: ValidationError) -> FormalismLoadError:
    """Translate pydantic's first error into the package's path-bearing FormalismLoadError,
    rendering the location in the `math_description.subdomains[0].kind` style used elsewhere.
    Discriminated-union members add their tag to the path (e.g. `boundary_conditions[0].neumann`)."""

    err = exc.errors()[0]
    path = "math_description"
    for part in err["loc"]:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    msg = err["msg"]
    # A ValueError raised in a schema BeforeValidator surfaces as "Value error, <msg>"; drop the prefix.
    prefix = "Value error, "
    if msg.startswith(prefix):
        msg = msg[len(prefix) :]
    return FormalismLoadError(path, msg)


def _read_text(source: str | Path) -> str:
    """Read text from a file path or pass through a string.

    A `Path` always means file path. A `str` is treated as a path only
    if it has no newline AND a file exists at that location; otherwise
    it's treated as literal source text.
    """

    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if "\n" not in source:
        candidate = Path(source)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return source
