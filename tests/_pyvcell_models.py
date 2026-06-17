"""Robustly load ``pyvcell.vcml.models_math`` for tests.

pyvcell's ``vcml`` package ``__init__`` is sometimes lazy (PEP 562) and sometimes eagerly
imports its heavy solver/viz/libvcell stack (depending on the working-tree state). The math
data model itself is pydantic-only, so when the direct import fails we load ``models_math.py``
directly, bypassing the package ``__init__``. Returns ``None`` only when pyvcell is not
installed at all (then the pyvcell-dependent tests skip).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def load_models_math() -> Any:
    """The ``pyvcell.vcml.models_math`` module, or ``None`` if pyvcell is not installed.
    Typed ``Any`` (not ``Any | None``) so call sites guarded by ``skipif(vm is None)`` need
    no per-attribute narrowing."""
    try:
        import pyvcell.vcml.models_math as vm  # works when vcml/__init__ is lazy

        return vm
    except ImportError:
        pass

    try:
        import pyvcell  # top-level __init__ is empty/light
    except ImportError:
        return None  # pyvcell not installed — pyvcell-dependent tests will skip

    if "pyvcell.vcml.models_math" in sys.modules:
        return sys.modules["pyvcell.vcml.models_math"]
    vcml_dir = Path(pyvcell.__file__).parent / "vcml"
    if "pyvcell.vcml" not in sys.modules:
        package = types.ModuleType("pyvcell.vcml")
        package.__path__ = [str(vcml_dir)]
        sys.modules["pyvcell.vcml"] = package
    spec = importlib.util.spec_from_file_location("pyvcell.vcml.models_math", vcml_dir / "models_math.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyvcell.vcml.models_math"] = module
    spec.loader.exec_module(module)
    return module
