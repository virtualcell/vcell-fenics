"""The backend's lazy exports survive their submodules being imported first.

``vcell_fenics.backend.assemble`` is both a submodule and the function it exports, and importing a submodule
binds it onto its package. When ``runner`` (which imports the submodule) loaded first, a later ``from
vcell_fenics.backend import assemble`` got the module: the test suite failed depending on file order. Run in a
fresh interpreter, so the import order is the test's own.
"""

from __future__ import annotations

import subprocess
import sys


def test_the_assemble_export_is_the_function_after_its_submodule_is_imported() -> None:
    script = (
        "import vcell_fenics.runner\n"  # imports vcell_fenics.backend.assemble first
        "import sys, types\n"
        "from vcell_fenics.backend import assemble\n"
        "from vcell_fenics.backend.assemble import rebuild_on_mesh\n"
        "assert callable(assemble) and not isinstance(assemble, types.ModuleType), assemble\n"
        "assert isinstance(sys.modules['vcell_fenics.backend.assemble'], types.ModuleType)\n"
        "assert callable(rebuild_on_mesh)\n"
        # a submodule that is not a lazy export of itself stays the module (its function `realize` is not
        # swapped in): `import … as m` reads the package attribute
        "import vcell_fenics.backend.realize as realize_module\n"
        "assert isinstance(realize_module, types.ModuleType), realize_module\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
