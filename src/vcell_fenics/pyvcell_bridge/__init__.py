"""Bridge from VCell's ``pyvcell`` to the vcell-fenics formalism.

The integration target of the project (CLAUDE.md): translate VCell models — the
``pyvcell.vcml.models_math.MathDescription`` (VCell's lowered reaction-diffusion math,
the focus of pyvcell's geom/math/bio split) — into the declarative formalism this
package solves with FEniCSx. See :mod:`~vcell_fenics.pyvcell_bridge.importer` for the
§2.6 mapping and :func:`~vcell_fenics.pyvcell_bridge.expression.translate_expression`
for the expression-syntax translation.

We depend on ``pyvcell.vcml.models_math`` only (a pure-Pydantic data model). pyvcell's
``vcml`` package is lazily imported (PEP 562), so this pulls none of pyvcell's heavy
solver/viz/binary closure into the DOLFINx env. pyvcell is installed editable
``--no-deps`` via the ``link-pyvcell`` task (it is not on PyPI, and its declared deps
would otherwise drag in that closure).
"""

from vcell_fenics.pyvcell_bridge.expression import translate_expression
from vcell_fenics.pyvcell_bridge.frame import normalize_to_geometry_frame
from vcell_fenics.pyvcell_bridge.geometry import import_geometry
from vcell_fenics.pyvcell_bridge.importer import (
    ImportResult,
    Observable,
    VcellImportError,
    import_math_description,
    import_model,
)

__all__ = [
    "ImportResult",
    "Observable",
    "VcellImportError",
    "import_geometry",
    "import_math_description",
    "import_model",
    "normalize_to_geometry_frame",
    "translate_expression",
]
