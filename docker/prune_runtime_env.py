"""Prune the runtime pixi environment in the image build: drop files the solver never loads.

Run by docker/Dockerfile (build stage, Linux only) with the environment's own Python, after the
real ``src/`` is in place:

    python docker/prune_runtime_env.py /app/.pixi/envs/default /app/src

Two prunes, both of things that are only ever *linked or compiled against*, never needed to run:

1. **VTK's unused shared libraries.** The PyPI ``vtk`` wheel is all of VTK (~640 MB: rendering,
   charts, widgets, IO for dozens of formats, viskores); the solver uses a data model, SurfaceNets
   and the XML writer. Every ``vtkmodules.<name>`` that ``src/`` imports is found by scanning the
   source, imported in a fresh interpreter, and every VTK shared object that import maps (the
   modules, their Python wrappers' dependencies, their transitive libraries) is kept, read from
   ``/proc/self/maps`` — what the loader actually loaded, not a hand-kept list. Every other ``.so``
   under ``vtkmodules/`` is deleted; the pure-Python files stay. A new ``vtkmodules`` import in
   ``src/`` is picked up automatically on the next build; the imports are re-checked in a second
   fresh interpreter after the prune, so a miss fails the build here rather than a user's run.
2. **C++ headers no runtime compile reads**: Boost's and OpenCASCADE's. DOLFINx/FFCx JIT-compile
   *C* kernels that include only ``ufcx.h`` and the C library; the C++ headers are there for building
   against DOLFINx (``libboost-devel`` is a run dependency of the conda package for CMake users) and
   Netgen. The compiler toolchain itself is kept.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

# C++ header trees, relative to the environment's include/, that no runtime JIT compile reads.
UNUSED_HEADER_DIRS = ("boost", "opencascade")

_IMPORT = re.compile(r"\bvtkmodules\.([A-Za-z_][\w.]*)")

_PROBE = """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
with open("/proc/self/maps") as maps:
    for line in maps:
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and "/vtkmodules/" in parts[5]:
            print(parts[5].strip())
"""


def vtk_modules_used(src: Path) -> list[str]:
    """Every ``vtkmodules.*`` module ``src/`` imports (``from vtkmodules.x.y import …`` → ``vtkmodules.x.y``)."""
    names = {f"vtkmodules.{m.rstrip('.')}" for p in src.rglob("*.py") for m in _IMPORT.findall(p.read_text())}
    return sorted(names)


def loaded_vtk_objects(python: Path, modules: list[str]) -> set[Path]:
    out = subprocess.run([str(python), "-c", _PROBE, *modules], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"importing {modules} failed (exit {out.returncode}):\n{out.stderr}")
    return {Path(line).resolve() for line in out.stdout.splitlines() if line}


def mib(n: int) -> str:
    return f"{n / 2**20:.0f} MiB"


def prune_vtk(env: Path, src: Path) -> None:
    python = env / "bin" / "python"
    # Ask the interpreter where vtkmodules is, rather than globbing lib/python3*: conda's python
    # also ships a lib/python3.1 → python3.12 compatibility symlink, which a glob sees twice.
    where = subprocess.run(
        [str(python), "-c", "import os, vtkmodules; print(os.path.dirname(vtkmodules.__file__))"],
        check=True,
        capture_output=True,
        text=True,
    )
    vtk_dir = Path(where.stdout.strip()).resolve()
    modules = vtk_modules_used(src)
    if not modules:
        raise SystemExit(f"no vtkmodules imports found under {src} — refusing to prune all of VTK")
    keep = loaded_vtk_objects(python, modules)
    removed = 0
    for so in vtk_dir.rglob("*.so*"):
        if so.is_file() and so.resolve() not in keep:
            removed += so.stat().st_size
            so.unlink()
    kept = sum(p.stat().st_size for p in keep if p.is_file())
    print(f"vtk: kept {len(keep)} shared objects ({mib(kept)}) for {', '.join(modules)}; removed {mib(removed)}")
    # The proof: the same imports, from scratch, against the pruned tree.
    loaded_vtk_objects(python, modules)


def prune_headers(env: Path) -> None:
    for name in UNUSED_HEADER_DIRS:
        tree = env / "include" / name
        if tree.is_dir():
            size = sum(p.stat().st_size for p in tree.rglob("*") if p.is_file())
            shutil.rmtree(tree)
            print(f"headers: removed include/{name} ({mib(size)})")


def main() -> None:
    env, src = (Path(a) for a in sys.argv[1:3])
    prune_vtk(env, src)
    prune_headers(env)


if __name__ == "__main__":
    main()
