#!/usr/bin/env bash
# Container entry point: activate the pixi environment, then run either the model runner or
# whatever command the caller asked for.
#
#   docker run … vcell-fenics --vcml model.vcml --out /work/out   # flags → the runner
#   docker run … vcell-fenics mpirun -n 4 python -m vcell_fenics.cli …   # a command → as given
#   docker run -it … vcell-fenics bash                            # a shell in the env
#
# The activation script is what `pixi shell-hook` produced at build time: it sets PATH,
# LD_LIBRARY_PATH and the conda activation variables (PETSc, HDF5, Netgen) that the
# environment needs. Sourcing it means the image carries no pixi binary of its own.
set -euo pipefail

# shellcheck disable=SC1091
source /opt/activate.sh

case "${1-}" in
    # No arguments at all: show what this image does rather than failing on a missing model.
    "") exec python -m vcell_fenics.cli --help ;;
    # Anything starting with a dash is a runner flag.
    -*) exec python -m vcell_fenics.cli "$@" ;;
    # Otherwise the caller named a command (bash, python, mpirun, pytest, …); run it in the env.
    *) exec "$@" ;;
esac
