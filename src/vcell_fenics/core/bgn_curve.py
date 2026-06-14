"""BGN tangential mesh redistribution for a curvature-driven membrane (1D case).

Mean-curvature flow `η v + σ H n = 0` (the surface-tension force balance, V = -m κ
with mobility `m = σ/η`) moves membrane nodes *purely normally*. Under pure normal
motion the parameterization is not maintained: where the curve contracts, nodes
crowd together; where it expands, they spread out. On a non-circular curve this
degrades the mesh — nodes bunch at the high-curvature tips until the discretization
tangles (the `MeshQualityError` the curvature-force path hits at large `dt`).

This module implements the **Barrett–Garcke–Nürnberg** (BGN) semi-implicit scheme,
whose defining feature is *intrinsic tangential redistribution*: it solves for the
new node positions and the curvature **together**, and — crucially — only pins each
node's *normal* velocity to the physical law, leaving the *tangential* position free
to be set by the curvature-vector identity. That spare tangential freedom is what
keeps the nodes asymptotically equidistributed in arc length, with no explicit
tangential velocity to tune.

The semidiscrete scheme on the *old* polyline Γᵐ (nodes Xᵐ, lumped mass M, 1D
stiffness A = ⟨∂ₛ·, ∂ₛ·⟩, unit nodal normals ν) solves for (Xᵐ⁺¹, κ):

    (B)   A Xᵈ − diag(M νᵈ) κ = 0           (d = each spatial component)
    (A)   Σᵈ diag(νᵈ) Xᵈ + m Δt κ = ν · Xᵐ   (normal velocity = −m κ)

(B) is the discrete curvature-vector identity ⟨κ ν, η⟩ = ⟨∂ₛ X, ∂ₛ η⟩ (the same
weak surface-Laplacian-of-position used by the curvature-force projection); (A)
imposes V·ν = −m κ. The geometry (ν, A, M) is from Γᵐ, so the system is **linear**
in (Xᵐ⁺¹, κ) — semi-implicit, unconditionally stable, and the tangential
redistribution falls out of (B) for free.

Pure NumPy / SciPy-sparse, operating on a closed ordered polyline (the membrane
loop). Driving an actual DOLFINx membrane `Mesh` with it — ordering the loop,
solving, writing the coordinates back — is the backend bridge (a separate concern,
like `surface_remap_mesh` sits on top of `surface_remap`).

Verification (`tests/test_core_bgn_curve.py`): a circle shrinks per the exact law
`r² = r₀² − 2 m t` and stays perfectly round; any convex curve loses area at the
constant curve-shortening rate `dA/dt = −2π m`; and on an ellipse the BGN edge-length
ratio stays bounded (mesh maintained) where a naive normal-only step bunches it an
order of magnitude worse.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

Floats = NDArray[np.float64]


def _loop_geometry(points: Floats) -> tuple[Floats, Floats, sp.csr_matrix, Floats]:
    """Per-node lumped mass `M`, edge lengths, 1D stiffness `A`, and unit nodal
    normals for a closed polyline, all on the polyline as given.

    `A` is the P1 Laplace–Beltrami stiffness ⟨∂ₛφᵢ, ∂ₛφⱼ⟩ assembled around the loop
    (each segment of length ℓ contributes (1/ℓ)[[1,−1],[−1,1]]). The nodal normal is
    the normalised average of the two adjacent edge normals; its *sign* is immaterial
    — the scheme is invariant under ν → −ν (κ absorbs it) — but it must be a unit
    vector (the scheme is not invariant to its magnitude).
    """

    n = len(points)
    nxt = np.roll(np.arange(n), -1)
    prv = np.roll(np.arange(n), 1)
    edge = points[nxt] - points  # eᵢ = X_{i+1} − Xᵢ (the closing edge wraps)
    elen: Floats = np.linalg.norm(edge, axis=1)
    if np.any(elen < 1e-14):
        raise ValueError("polyline has a zero-length (coincident-node) edge")

    mass: Floats = 0.5 * (elen + elen[prv])  # lumped: half of each adjacent edge

    # Stiffness: edge i (weight wᵢ = 1/ℓᵢ) couples nodes i and nxt[i]. Diagonal node i
    # gets wᵢ (edge i) + w_{i−1} (edge i−1); off-diagonals are −wᵢ both ways.
    w: Floats = 1.0 / elen
    diag = w + w[prv]
    rows = np.concatenate([np.arange(n), nxt, np.arange(n)])
    cols = np.concatenate([np.arange(n), np.arange(n), nxt])
    vals = np.concatenate([diag, -w, -w])
    stiffness = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    tangent = edge / elen[:, None]
    edge_normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))  # ⊥ to each edge
    nodal = edge_normal + edge_normal[prv]
    nodal /= np.linalg.norm(nodal, axis=1)[:, None]
    return mass, elen, stiffness, nodal


def bgn_curvature_flow_step(points: Floats, *, mobility: float, dt: float) -> Floats:
    """One BGN semi-implicit step of mean-curvature flow on a closed polyline.

    `points` is an (N, 2) array of the membrane's nodes in loop order (the closing
    edge runs from the last node back to the first). `mobility` is `m = σ/η` from the
    surface-tension force balance `η v + σ H n = 0` (so the normal velocity is
    `V = −m κ`, inward for a convex curve). Returns the new (N, 2) node positions:
    the curve has advanced by curvature flow *and* the nodes have been tangentially
    redistributed, in one coupled linear solve on the old geometry.
    """

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("points must be an (N, 2) array with N >= 3 nodes (a closed 2D loop)")
    if mobility <= 0.0 or dt <= 0.0:
        raise ValueError("mobility and dt must be positive")

    n = len(points)
    mass, _elen, stiffness, nu = _loop_geometry(points)
    nu0, nu1 = nu[:, 0], nu[:, 1]
    zero = sp.csr_matrix((n, n))

    # Unknown vector u = [X0 (N), X1 (N), κ (N)]. See module docstring for (A)/(B).
    block_x0 = sp.hstack([stiffness, zero, sp.diags(-mass * nu0)])  # (B) component 0
    block_x1 = sp.hstack([zero, stiffness, sp.diags(-mass * nu1)])  # (B) component 1
    block_n = sp.hstack([sp.diags(nu0), sp.diags(nu1), (mobility * dt) * sp.eye(n)])  # (A)
    system = sp.vstack([block_x0, block_x1, block_n]).tocsr()

    rhs = np.concatenate([np.zeros(n), np.zeros(n), nu0 * points[:, 0] + nu1 * points[:, 1]])
    solution = spla.spsolve(system, rhs)
    return np.column_stack([solution[:n], solution[n : 2 * n]])


def polygon_area(points: Floats) -> float:
    """Signed area enclosed by a closed polyline (shoelace formula).

    Positive for a counter-clockwise loop. Used to verify the curve-shortening area
    law `dA/dt = −2π·mobility` (independent of the convex curve's shape).
    """

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("points must be an (N, 2) array with N >= 3 nodes")
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
