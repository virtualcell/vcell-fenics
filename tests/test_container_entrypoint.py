"""The container entry point (`docker/entrypoint.sh`), exercised without Docker.

Apptainer/Singularity runs the image read-only, so the pre-warmed FFCx cache at /opt/cache cannot take
new kernels there; the entry point must notice and move to a writable per-user cache seeded from the
pre-warmed one (ADR 011 §5). `VCELL_FENICS_ACTIVATE` lets the script run here against an empty
activation file; the real image is covered by the Docker/Apptainer smoke jobs in CI.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh"
_SHOW = ["sh", "-c", 'printf "%s|%s" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"']


def _run(tmp_path: Path, argv: list[str], **env: str) -> subprocess.CompletedProcess[str]:
    activate = tmp_path / "activate.sh"
    # Like the image's real activation (pixi shell-hook), which sources conda's bash-completion files:
    # hwloc's reads $ZSH_VERSION unguarded — under the entry point's `set -u` that aborted the container.
    activate.write_text('if [ -n "$ZSH_VERSION" ]; then :; fi\nexport VCELL_FENICS_ACTIVATED=1\n')
    environment = {
        k: v for k, v in os.environ.items() if k not in ("XDG_CACHE_HOME", "MPLCONFIGDIR", "VCELL_FENICS_CACHE")
    }
    environment |= {"VCELL_FENICS_ACTIVATE": str(activate), "TMPDIR": str(tmp_path / "tmp"), **env}
    (tmp_path / "tmp").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(_ENTRYPOINT), *argv], capture_output=True, text=True, env=environment, timeout=120
    )


def test_a_writable_cache_is_used_as_is(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    result = _run(tmp_path, _SHOW, XDG_CACHE_HOME=str(cache))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{cache}|{cache}/matplotlib"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through read-only permissions")
def test_a_read_only_cache_moves_to_a_seeded_writable_one(tmp_path: Path) -> None:
    image_cache = tmp_path / "opt-cache"
    (image_cache / "fenics").mkdir(parents=True)
    (image_cache / "fenics" / "prewarmed_kernel.so").write_text("kernel")
    for path in (image_cache / "fenics" / "prewarmed_kernel.so", image_cache / "fenics", image_cache):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))  # a SIF is read-only
    try:
        result = _run(tmp_path, _SHOW, XDG_CACHE_HOME=str(image_cache))
    finally:
        for path in (image_cache, image_cache / "fenics"):
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    assert result.returncode == 0, result.stderr
    writable = tmp_path / "tmp" / f"vcell-fenics-cache-{os.getuid()}"
    assert result.stdout == f"{writable}|{writable}/matplotlib"
    assert (writable / "fenics" / "prewarmed_kernel.so").read_text() == "kernel"  # seeded


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through read-only permissions")
def test_vcell_fenics_cache_names_a_shared_cache(tmp_path: Path) -> None:
    image_cache = tmp_path / "opt-cache"
    image_cache.mkdir()
    image_cache.chmod(0o555)
    shared = tmp_path / "shared-cache"
    try:
        result = _run(tmp_path, _SHOW, XDG_CACHE_HOME=str(image_cache), VCELL_FENICS_CACHE=str(shared))
    finally:
        image_cache.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"{shared}|")


def test_an_activation_script_reading_unset_variables_does_not_abort(tmp_path: Path) -> None:
    result = _run(tmp_path, ["sh", "-c", 'printf "%s" "$VCELL_FENICS_ACTIVATED"'], XDG_CACHE_HOME=str(tmp_path / "c"))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "1"


def test_flags_go_to_the_runner(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--help"], XDG_CACHE_HOME=str(tmp_path / "cache"))
    assert result.returncode == 0, result.stderr
    assert "--simtask" in result.stdout and "--vc-print-status" in result.stdout
