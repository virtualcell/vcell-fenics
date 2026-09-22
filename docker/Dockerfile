# syntax=docker/dockerfile:1.7
#
# vcell-fenics runner image: a VCell (.vcml / pyvcell YAML) or native-formalism model in,
# solved fields out into a mounted directory. Build from the repository root:
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
# something to install against. This keeps the expensive layer (~2 GB of conda packages) keyed
# on the lock alone: editing solver code rebuilds in seconds, not minutes.
COPY pyproject.toml pixi.lock README.md LICENSE.md ./
RUN mkdir -p src/vcell_fenics && touch src/vcell_fenics/__init__.py src/vcell_fenics/py.typed

# The cache mount keeps downloaded packages across builds; `--locked` fails loudly if
# pyproject.toml and pixi.lock have drifted apart, rather than silently re-solving.
#
# `--concurrent-downloads 8` (pixi defaults to 50): this environment is ~700 packages and
# several GB, and 50 simultaneous connections is enough to trip connection limits on corporate
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

# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime

# No apt layer on purpose. The pixi environment is self-contained down to the X/GL shared
# objects VTK links against (conda-forge ships libglvnd/libgl/xorg-libx11 as dependencies of
# pyvista), so the base image supplies nothing but glibc and the build needs no package
# repository beyond conda-forge.

COPY --from=build /opt/activate.sh /opt/activate.sh
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/src /app/src
COPY --from=build /app/pyproject.toml /app/pyproject.toml
COPY docker/entrypoint.sh /usr/local/bin/vcell-fenics-entrypoint
COPY examples/models /opt/vcell-fenics/examples

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
    # HDF5 file locking fails on many bind-mounted volumes (macOS/Windows shares, NFS); the
    # XDMF writer would abort on `fields.h5` at the first output.
    HDF5_USE_FILE_LOCKING=FALSE \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

# Warm the JIT cache and smoke-test the image in one step: if the demo model does not solve,
# the build fails here rather than in the user's first `docker run`.
RUN mkdir -p /opt/cache \
 && /usr/local/bin/vcell-fenics-entrypoint \
      --math /opt/vcell-fenics/examples/diffusion2d_math.yaml \
      --geometry /opt/vcell-fenics/examples/diffusion2d_geom.yaml \
      --t-final 0.1 --dt 0.05 --h 0.4 --out /tmp/warmup \
 && rm -rf /tmp/warmup \
 && chmod -R a+rwX /opt/cache

# A default non-root identity. On Linux hosts, run with `--user "$(id -u):$(id -g)"` instead so
# the files written into the mounted results directory belong to you; nothing in the image
# depends on this particular uid — which is why /work is world-writable rather than owned by
# it: an un-mounted run must still be able to create ./out under any uid.
RUN useradd --create-home --uid 1001 vcell \
 && mkdir -p /work \
 && chmod 1777 /work
USER 1001:1001

# Results land in ./out by default, so a bind mount at /work/out is all a run needs.
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/vcell-fenics-entrypoint"]
CMD ["--help"]
