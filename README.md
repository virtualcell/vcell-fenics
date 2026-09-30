# vcell-fenics

FEniCSx (DOLFINx) backend for Virtual Cell models and a VCell solver: a VCell SimulationTask in, a
VTU + zarr results bundle out. Orientation: `docs/overview.md`; the container: `docker/README.md`; the
release contract: `SOLVER-RELEASE.md`.

## Releases

A release is a tag `vX.Y.Z` on `main`; the artifacts are the container images (no archives).

1. In a PR, set the version in **both** `pyproject.toml` (`[project].version`) and
   `src/vcell_fenics/__init__.py` (`__version__`) — `tests/test_version.py` checks they agree — and merge it.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git fetch origin && git tag -a vX.Y.Z origin/main -m "vcell-fenics X.Y.Z" && git push origin vX.Y.Z
   ```

3. The `container` workflow runs on the tag. It fails at once if the tag is not `v<package version>`
   or not on `main`, builds and smoke-tests the image and SIF (checking both report `X.Y.Z`), then
   publishes `ghcr.io/virtualcell/vcell-fenics:X.Y.Z` (and `:vX.Y.Z`, multi-arch) and
   `oras://ghcr.io/virtualcell/vcell-fenics_singularity:X.Y.Z` (and `:vX.Y.Z`, amd64), and creates the
   GitHub Release with notes. `latest` keeps following `main`.

A mistaken tag is fixed by bumping the version and tagging again — never by moving a published tag.
