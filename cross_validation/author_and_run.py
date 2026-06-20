"""Author minimal 2D reaction-diffusion VCell biomodels, run VCell's finite-volume
solver (via pyvcell), and return the 4D solution field on the solver's own grid.

This lets vcell-fenics author VCell models and run the FV solver itself, then
cross-validate its FEniCSx solver against the FV field point-by-point.

Requires pyvcell installed with the `native` + `solver` extras
(libvcell + pyvcell-fvsolver):

    pip install "pyvcell[native,solver]"      # or pyvcell[all]

Quick use:

    from author_and_run import author_and_run_diffusion_2d
    out = author_and_run_diffusion_2d(
        ic_expr="(10.0 * exp( - (pow(x - 0.3, 2.0) + pow(y, 2.0)) / 0.02))",
        diffusion=0.1, decay_k=0.0,
    )
    out["field"]   # u(t, c, y, x), float64, shape (T, C, Y, X)
    out["x"], out["y"], out["t"]   # the solver's grid coordinates
    out["vcml"]    # the exact VCML bytes the FV solver consumed

Run as a script to (re)generate the v1 / v1b / v2 reference datasets next to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import pyvcell.vcml as vc
from pyvcell.vcml.models import Biomodel, Model
from pyvcell.vcml.utils import to_vcml_str


def author_diffusion_2d(
    *,
    ic_expr: str,
    diffusion: float = 0.1,
    decay_k: float = 0.0,
    extent: tuple[float, float, float] = (2.0, 2.0, 1.0),
    origin: tuple[float, float, float] = (-1.0, -1.0, 0.0),
    mesh: tuple[int, int, int] = (128, 128, 1),
    end_time: float = 1.0,
    output_dt: float = 0.1,
    biomodel_name: str = "Diffusion2D",
    application: str = "diffusion2D",
    simulation: str = "diffusion_sim",
) -> Biomodel:
    """Build a single-species, single-compartment 2D reaction-diffusion biomodel.

    The whole rectangular domain is one analytic subvolume (`1.0`), species ``u``
    diffuses with constant `diffusion`, boundaries are no-flux (zero-Neumann), and
    an optional first-order decay ``-decay_k * u`` is added as a mass-action reaction.

    Args:
        ic_expr: VCell expression for the initial concentration of ``u`` (uses
            spatial vars ``x``, ``y``, ``z`` — e.g. an off-center Gaussian).
        diffusion: isotropic diffusion coefficient D.
        decay_k: first-order decay rate; 0.0 for pure diffusion.
        extent/origin: geometry box (z is a unit-thickness slab for 2D).
        mesh: FV element counts (nx, ny, 1) for 2D.
        end_time/output_dt: simulation duration and output interval.
    """
    geo = vc.Geometry(name="square", dim=2, extent=extent, origin=origin)
    geo.add_background("domain")  # analytic expression "1.0" -> the whole box

    model = Model(name="diffusion")
    model.add_compartment("cell", dim=3)
    model.add_species("u", "cell")
    if decay_k:
        model.add_reaction_mass_action("decay", "cell", reactants=["u"], products=[], kf=decay_k, kr=0.0)

    biomodel = Biomodel(name=biomodel_name, model=model)
    app = biomodel.add_application(application, geometry=geo)
    app.map_compartment("cell", "domain")
    app.map_species("u", init_conc=ic_expr, diff_coef=diffusion)
    if decay_k:
        app.map_reaction("decay", True)
    app.add_sim(name=simulation, duration=end_time, output_time_step=output_dt, mesh_size=mesh)
    return biomodel


def result_to_4d(result: Any) -> dict[str, Any]:
    """Extract the 4D solution field from a pyvcell ``Result``.

    Returns a dict:
      - ``field``    (T, C, Y, X) float64 — the volume-variable (species) fields,
                     in the same order as ``channels``;
      - ``channels`` list[str] — species names (length C);
      - ``t``, ``x``, ``y`` — 1D coordinate arrays. ``x``/``y`` are the FV solver's
        own cell coordinates, so another solver should sample at exactly these
        points; ``field[ti, ci, j, i]`` is species ``channels[ci]`` at ``(x[i], y[j])``,
        time ``t[ti]``.
    """
    zd = result.zarr_dataset  # (T, C_all, Z, Y, X) — includes coordinate/derived channels
    index_by_label = {c.label: c.index for c in result.channel_data}
    t = np.asarray(result.time_points, dtype=float)
    x = np.asarray(zd[0, index_by_label["x"], 0, 0, :], dtype=float)
    y = np.asarray(zd[0, index_by_label["y"], 0, :, 0], dtype=float)

    channels = [name.split("::")[-1] for name in result.volume_variable_names]
    field = np.stack(
        [np.asarray(zd[:, index_by_label[name], 0, :, :], dtype=float) for name in channels],
        axis=1,
    )  # (T, C, Y, X)
    return {"field": field, "channels": channels, "t": t, "x": x, "y": y}


def author_and_run_diffusion_2d(**kwargs: Any) -> dict[str, Any]:
    """Author the biomodel, run VCell's FV solver, and return the 4D result.

    Accepts the same keyword args as :func:`author_diffusion_2d`. Returns the
    :func:`result_to_4d` dict plus ``vcml`` (the exact bytes the solver consumed),
    ``biomodel``, ``application`` and ``simulation`` names.
    """
    biomodel = author_diffusion_2d(**kwargs)
    vcml = to_vcml_str(biomodel)  # canonical VCML the FV solver actually runs
    application = biomodel.applications[0].name
    simulation = biomodel.applications[0].simulations[0].name

    result = vc.simulate(vc.load_vcml_str(vcml), simulation)
    try:
        out = result_to_4d(result)
    finally:
        result.cleanup()
    out.update(vcml=vcml, biomodel=biomodel.name, application=application, simulation=simulation)
    return out


def free_space_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    amplitude: float = 10.0,
    a: float = 0.02,
    diffusion: float = 0.1,
    center: tuple[float, float] = (0.3, 0.0),
    decay_k: float = 0.0,
) -> np.ndarray:
    """Free-space analytic solution for a Gaussian IC under diffusion (+ decay).

    For ``u_t = D∇²u - k·u`` with ``u(·,0) = A·exp(-r²/a)`` on an unbounded domain:
    ``u = e^{-k t} · A · a/(a+4Dt) · exp(-r²/(a+4Dt))``. Valid until the front
    reaches a wall; an independent reference for both solvers at early/mid times.
    Returns shape (T, Y, X).
    """
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (Y, X); [j, i] = (x[i], y[j])
    out = np.empty((len(t), len(y), len(x)), dtype=float)
    for ti, tt in enumerate(t):
        s = a + 4.0 * diffusion * float(tt)
        out[ti] = np.exp(-decay_k * float(tt)) * amplitude * (a / s) * np.exp(
            -((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / s
        )
    return out


# Off-center Gaussian IC, parameterized by width `a` (so a=0.02 -> /0.02).
def _gaussian_ic(a: float) -> str:
    return f"(10.0 * exp( - (pow(x - 0.3, 2.0) + pow(y, 2.0)) / {a}))"


_MODELS = [
    ("minimal_diffusion_2d_v1", dict(ic_expr=_gaussian_ic(0.02), biomodel_name="MinimalDiffusion2D"), 0.02, 0.0),
    ("minimal_diffusion_2d_v1b", dict(ic_expr=_gaussian_ic(0.05), biomodel_name="MinimalDiffusion2D_broadIC"), 0.05, 0.0),
    (
        "minimal_diffusion_2d_v2",
        dict(ic_expr=_gaussian_ic(0.02), biomodel_name="MinimalDiffusion2D_decay", decay_k=0.5),
        0.02,
        0.5,
    ),
]


def main() -> None:
    """(Re)generate the v1 / v1b / v2 reference .vcml + .npz next to this script."""
    here = Path(__file__).parent
    for stem, kwargs, a, decay_k in _MODELS:
        out = author_and_run_diffusion_2d(**kwargs)
        (here / f"{stem}.vcml").write_text(out["vcml"])
        analytic = free_space_gaussian(out["x"], out["y"], out["t"], a=a, decay_k=decay_k)[:, None, :, :]
        np.savez_compressed(
            here / f"{stem}_reference.npz",
            field=out["field"],
            analytic=analytic,
            t=out["t"],
            x=out["x"],
            y=out["y"],
            channels=np.array(out["channels"]),
            axes=np.array(["t", "c", "y", "x"]),
            a=a,
            decay_k=decay_k,
        )
        total = out["field"][:, 0].reshape(len(out["t"]), -1).sum(1) * (out["x"][1] - out["x"][0]) * (
            out["y"][1] - out["y"][0]
        )
        print(
            f"{stem}: field={out['field'].shape} channels={out['channels']} "
            f"app={out['application']} sim={out['simulation']} "
            f"total[0]={total[0]:.4f} total[-1]={total[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
