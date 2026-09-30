"""The package version has two homes — `[project].version` and `vcell_fenics.__version__` — and a release
tag `vX.Y.Z` must equal both (SOLVER-RELEASE.md; the container workflow checks the tag against the image)."""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import vcell_fenics

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pyproject_and_package_versions_agree() -> None:
    project = tomllib.loads(_PYPROJECT.read_text())["project"]
    assert project["version"] == vcell_fenics.__version__


def test_the_version_is_plain_semver() -> None:
    # A tag vX.Y.Z publishes image tags X.Y.Z and vX.Y.Z; anything else would not match the tag pattern.
    assert re.fullmatch(r"\d+\.\d+\.\d+", vcell_fenics.__version__)


def test_installed_metadata_matches() -> None:
    # The editable install's metadata is written from pyproject at install time; a bump without
    # re-installing leaves it stale (run `pixi install`).
    assert version("vcell-fenics") == vcell_fenics.__version__
