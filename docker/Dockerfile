# syntax=docker/dockerfile:1.7
#
# vcell-fenics runner image: a VCell SimulationTask / .vcml / pyvcell YAML or native-formalism model
# in, a results bundle out into a mounted directory (ADR 010/011). Also the source of the Apptainer
# SIF VCell's cluster runs (.github/workflows/container.yml). Build from the repository root:
#
#     docker build -f docker/Dockerfile -t vcell-fenics .
#
# The environment is the project's own `pixi.lock`, installed with `--locked`, so the
# container solves with exactly the DOLFINx/PETSc/Netgen builds the test suite runs against —
# that is the point of building from the lock rather than from a `conda install` line.
#
# Platforms: the lock carries linux-64 and linux-aarch64, so this builds natively on both
# x86-64 and Apple-silicon/Graviton hosts. Adding a platform means editing
# `[tool.pixi.workspace].platforms` and re-running `pixi lock`.

ARG PIXI_VERSION=0.68.0
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Stage 1 — resolve and install the pixi environment.
# ---------------------------------------------------------------------------
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}-noble AS build

WORKDIR /app

# Prefer IPv4 during the build. conda.anaconda.org publishes both A and AAAA records, and a
# Docker VM with no working IPv6 route (Docker Desktop's default on some hosts) resolves the
# AAAA addresses fine and then *hangs* on every connect to one until it times out — which is
# what a late, random `failed to fetch <package> … tcp connect error` in the install below
# actually is. This nudges glibc's getaddrinfo to sort IPv4 first. If a build still stalls that
# way, turn IPv6 off for the VM itself (Docker Desktop → Settings → Resources → Network).
RUN printf 'precedence ::ffff:0:0/96  100\n' >> /etc/gai.conf

# Manifest + lock only, plus a stub package so the editable `vcell-fenics` pypi-dependency has
# something to install against. This keeps the expensive layer (the conda packages and PyPI
# wheels) keyed on the lock alone: editing solver code rebuilds in seconds, not minutes.
COPY pyproject.toml pixi.lock README.md LICENSE.md ./
RUN mkdir -p src/vcell_fenics && touch src/vcell_fenics/__init__.py src/vcell_fenics/py.typed

# The cache mount keeps downloaded packages across builds; `--locked` fails loudly if
# pyproject.toml and pixi.lock have drifted apart, rather than silently re-solving.
#
# `--concurrent-downloads 8` (pixi defaults to 50): this environment is ~260 packages and
# ~2 GB, and 50 simultaneous connections is enough to trip connection limits on corporate
# NAT/proxies and on Docker Desktop's VM network — which surfaces as a *late* `tcp connect
# error` partway through, not as an immediate failure. The retry loop then resumes from the
# cache mount rather than re-downloading, and a genuinely broken build still fails after the
# last attempt.
ARG PIXI_DOWNLOAD_RETRIES=4
RUN --mount=type=cache,target=/root/.cache/rattler,sharing=locked \
    for attempt in $(seq 1 "${PIXI_DOWNLOAD_RETRIES}"); do \
        if pixi install --locked --environment default --concurrent-downloads 8; then exit 0; fi; \
        echo "pixi install attempt ${attempt} failed; retrying in 15s" >&2; \
        sleep 15; \
    done; \
    exit 1

# The activation script (PATH, LD_LIBRARY_PATH, and the conda activation hooks PETSc/HDF5/
# Netgen rely on), so the runtime stage needs no pixi binary at all.
RUN pixi shell-hook --environment default --shell bash > /opt/activate.sh

# The real source, over the stub. The editable install resolves through /app/src at import
# time, so replacing the tree here is all that is needed — no reinstall.
COPY src/ ./src/

# Drop what the solver never loads: the VTK shared libraries outside what src/'s vtkmodules
# imports actually map (the PyPI wheel is all of VTK; we use SurfaceNets and the XML writer), and
# the Boost/OpenCASCADE C++ headers (FFCx's JIT compiles C kernels that never include them; the
# compiler toolchain stays). The script re-imports every VTK module src/ uses after pruning, so a
# miss fails here. It runs after `COPY src/` because the VTK keep-list is derived from the source.
COPY docker/prune_runtime_env.py /tmp/prune_runtime_env.py
RUN /app/.pixi/envs/default/bin/python /tmp/prune_runtime_env.py /app/.pixi/envs/default /app/src \
 && du -sh /app/.pixi/envs/default

# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime

# No apt layer on purpose. The pixi environment is self-contained: VTK is the PyPI wheel, which
# links nothing beyond glibc/libstdc++ (no X, GL or Qt — it would dlopen those only to render,
# which the solver never does), so the base image supplies nothing but glibc and the build needs no
# package repository beyond conda-forge and PyPI.

COPY --from=build /opt/activate.sh /opt/activate.sh
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/src /app/src
COPY --from=build /app/pyproject.toml /app/pyproject.toml
COPY docker/entrypoint.sh /usr/local/bin/vcell-fenics-entrypoint
COPY examples/models /opt/vcell-fenics/examples
# Real VCell SimulationTask documents: the fvsolver smoke task is the image's VCell-path self-test.
COPY tests/fixtures/simtask /opt/vcell-fenics/examples/simtask
COPY tests/fixtures/image /opt/vcell-fenics/examples/image

# FFCx/DOLFINx JIT-compile every form on first use and cache the generated kernels under
# $XDG_CACHE_HOME/fenics. In a container that directory must (a) exist outside $HOME, since the
# image is meant to be run with an arbitrary `--user`, and (b) be world-writable, or every run
# pays the compile cost and a read-only home makes it fail outright. The warm-up below
# pre-populates it with the common P1 triangle/tetrahedron kernels.
ENV XDG_CACHE_HOME=/opt/cache \
    HOME=/tmp \
    # One thread per rank: DOLFINx/PETSc parallelism is MPI, and letting OpenMP also fan out
    # oversubscribes the cores and slows a `mpirun -n N` run down.
    OMP_NUM_THREADS=1 \
    # HDF5 file locking fails on many bind-mounted volumes (macOS/Windows shares, NFS); keep it
    # off for anything that still writes HDF5 (the XDMF export, h5py-based tools).
    HDF5_USE_FILE_LOCKING=FALSE \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

# Warm the JIT cache and smoke-test the image in one step: if either path does not solve, the
# build fails here rather than in the user's first `docker run`.
#   1. the native demo model (backward Euler);
#   2. the VCell path — a real SimulationTask, method of lines, with VCell's stdout status markers
#      (stdout must carry markers only; the check below fails on anything else) — and the bundle
#      reader and export on its result.
#   3. a VCell moving-boundary SimulationTask (the ALE path: a moving mesh, backward Euler, a
#      segmented bundle) — which also caches the moving problem's forms.
#   4. a real moving-boundary BioModel (the cleavage furrow) run long enough to remesh, so the
#      conservative remap runs against this image's own NumPy (NumPy 2.5 removed 2D np.cross, which
#      broke every remesh here while the osx-arm64 lock still had 2.4) — it must open a second segment.
#   5. a model on a real segmented 3D image (VCell's tutorial geometry: label field, SurfaceNets, a
#      Netgen volume mesh from the conforming surfaces) solved by the two-compartment coupled solver.
RUN mkdir -p /opt/cache \
 && /usr/local/bin/vcell-fenics-entrypoint \
      --math /opt/vcell-fenics/examples/diffusion2d_math.yaml \
      --geometry /opt/vcell-fenics/examples/diffusion2d_geom.yaml \
      --t-final 0.1 --dt 0.05 --h 0.4 --out /tmp/warmup \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics \
      --simtask /opt/vcell-fenics/examples/simtask/SimID_1585623750_0__0.simtask.xml \
      --out /tmp/warmup --vc-print-status > /tmp/warmup/markers.txt \
 && grep -q '\[\[\[progress:100.0%\]\]\]' /tmp/warmup/markers.txt \
 && ! grep -v '^\[\[\[.*\]\]\]$' /tmp/warmup/markers.txt \
 && /usr/local/bin/vcell-fenics-entrypoint python -m vcell_fenics.results \
      /tmp/warmup/SimID_1585623750_0_.fenics --require-status completed \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics-export \
      /tmp/warmup/SimID_1585623750_0_.fenics /tmp/warmup/paraview \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics \
      --simtask /opt/vcell-fenics/examples/simtask/SimID_274641196_0__0.simtask.xml \
      --out /tmp/warmup --vc-print-status > /tmp/warmup/mb-markers.txt \
 && grep -q '\[\[\[progress:100.0%\]\]\]' /tmp/warmup/mb-markers.txt \
 && /usr/local/bin/vcell-fenics-entrypoint python -m vcell_fenics.results \
      /tmp/warmup/SimID_274641196_0_.fenics --require-status completed \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics-export \
      /tmp/warmup/SimID_274641196_0_.fenics /tmp/warmup/paraview-mb \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics \
      --simtask /opt/vcell-fenics/examples/simtask/furrow_SimID_1486629996_0__0.simtask.xml \
      --out /tmp/warmup --h 0.3 --t-final 3 \
 && test -d /tmp/warmup/SimID_1486629996_0_.fenics/seg0001 \
 && /usr/local/bin/vcell-fenics-entrypoint vcell-fenics \
      --math /opt/vcell-fenics/examples/image/nuclear_math.yaml \
      --geometry /opt/vcell-fenics/examples/image/tutorial_geom.yaml \
      --t-final 0.25 --output-dt 0.25 --h 2.5 --out /tmp/warmup/image \
 && /usr/local/bin/vcell-fenics-entrypoint python -m vcell_fenics.results \
      /tmp/warmup/image/results.fenics --require-status completed \
 && rm -rf /tmp/warmup \
 && chmod -R a+rwX /opt/cache

# A default non-root identity. On Linux hosts, run with `--user "$(id -u):$(id -g)"` instead so
# the files written into the mounted results directory belong to you; nothing in the image
# depends on this particular uid — which is why /work is world-writable rather than owned by
# it: an un-mounted run must still be able to create ./out under any uid.
# /usr/local/bin/vcell-solver-entrypoint is the entry point's name in every VCell solver image
# (SOLVER-RELEASE.md); vcell-fenics-entrypoint stays for anything that already calls it.
RUN useradd --create-home --uid 1001 vcell \
 && ln -s vcell-fenics-entrypoint /usr/local/bin/vcell-solver-entrypoint \
 && mkdir -p /work \
 && chmod 1777 /work
USER 1001:1001

# Results land in ./out by default, so a bind mount at /work/out is all a run needs.
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/vcell-solver-entrypoint"]
CMD ["--help"]
