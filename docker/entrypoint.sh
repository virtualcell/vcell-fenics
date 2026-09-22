#!/usr/bin/env bash
# Container entry point: activate the pixi environment, then run either the model runner or
# whatever command the caller asked for.
#
#   docker run … vcell-fenics --simtask /data/SimID_1_0__0.simtask.xml   # flags → the runner
#   docker run … vcell-fenics vcell-fenics --simtask …                   # VCell's HPC form (SlurmProxy)
#   docker run … vcell-fenics mpirun -n 4 vcell-fenics --simtask …       # a command → as given
#   docker run -it … vcell-fenics bash                                   # a shell in the env
#
# The activation script is what `pixi shell-hook` produced at build time: it sets PATH,
# LD_LIBRARY_PATH and the conda activation variables (PETSc, HDF5, Netgen) that the
# environment needs. Sourcing it means the image carries no pixi binary of its own.
# (VCELL_FENICS_ACTIVATE overrides its path — for testing this script outside the image.)
set -euo pipefail

# nounset off while activating: conda's activation pulls in bash-completion scripts (hwloc's among
# them) that read unset variables such as $ZSH_VERSION, and under `set -u` the source would abort
# the container before it runs anything.
set +u
# shellcheck disable=SC1090
source "${VCELL_FENICS_ACTIVATE:-/opt/activate.sh}"
set -u

# FFCx compiles each new form into $XDG_CACHE_HOME/fenics. The image pre-warms that cache in a
# world-writable /opt/cache, but Apptainer/Singularity runs the image *read-only* (a SIF is a
# squashfs), so a form the warm-up did not cover would fail to compile. When the cache is not
# writable, move to a per-user one — $VCELL_FENICS_CACHE, else under $TMPDIR — seeded with the
# pre-warmed kernels. Point VCELL_FENICS_CACHE at a shared directory to keep kernels across jobs.
cache="${XDG_CACHE_HOME:-/opt/cache}"
if ! { mkdir -p "${cache}/fenics" && touch "${cache}/fenics/.writable"; } 2>/dev/null; then
    writable="${VCELL_FENICS_CACHE:-${TMPDIR:-/tmp}/vcell-fenics-cache-$(id -u)}"
    mkdir -p "${writable}"
    if [ -d "${cache}/fenics" ]; then
        cp -R -n "${cache}/fenics" "${writable}/" 2>/dev/null || true
    fi
    export XDG_CACHE_HOME="${writable}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME:-/opt/cache}/matplotlib}"

case "${1-}" in
    # No arguments at all: show what this image does rather than failing on a missing model.
    "") exec python -m vcell_fenics.cli --help ;;
    # Anything starting with a dash is a runner flag.
    -*) exec python -m vcell_fenics.cli "$@" ;;
    # Otherwise the caller named a command (vcell-fenics, vcell-fenics-export, bash, python,
    # mpirun, pytest, …); run it in the env.
    *) exec "$@" ;;
esac
