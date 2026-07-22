"""2-D node layout from the LRG diffusion distance.

The Laplacian Renormalisation Group induces, at a diffusion scale ``tau``, a
distance between nodes: the symmetrised inverse of the diffusion kernel
``rho(tau)`` (nodes that diffusion keeps well-connected are close). Embedding
that distance into the plane with metric MDS gives a layout where LRG
communities sit together — the "LRG-distance-inspired" arrangement used by the
dashboard's clustered-network view, in contrast to a generic spring layout that
ignores the diffusion geometry.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .kernel import graph_laplacian_and_spectrum, rho_matrix

__all__ = ["diffusion_distance_matrix", "diffusion_distance_mds"]


def diffusion_distance_matrix(
    graph: nx.Graph, tau: float, *, normalized: bool = True
) -> np.ndarray:
    """Square symmetric LRG diffusion-distance matrix at scale ``tau``.

    ``d_ij = max(1/rho_ij, 1/rho_ji)`` with a zero diagonal, where ``rho`` is the
    normalised diffusion kernel ``exp(-tau L) / Tr(...)``. Node order matches
    ``graph.nodes()``.
    """
    L, _spectrum = graph_laplacian_and_spectrum(
        graph, weight="weight", normalized=normalized
    )
    rho = rho_matrix(tau, L)
    with np.errstate(divide="ignore"):
        inv = 1.0 / rho
    dist = np.maximum(inv, inv.T)
    # A disconnected graph has a block-diagonal diffusion kernel, so cross-component
    # rho entries are exactly 0 → 1/rho = inf. Those pairs are *maximally* far (no
    # diffusion path), NOT coincident: map inf/nan to a large finite value (well
    # beyond the largest real distance) so MDS pushes components apart instead of
    # collapsing them onto each other.
    finite = dist[np.isfinite(dist)]
    big = (float(finite.max()) * 2.0) if finite.size and finite.max() > 0 else 1.0
    dist[~np.isfinite(dist)] = big
    np.fill_diagonal(dist, 0.0)
    return dist


def diffusion_distance_mds(
    graph: nx.Graph,
    tau: float,
    *,
    normalized: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """``(n_nodes, 2)`` MDS embedding of the LRG diffusion distance at ``tau``.

    Node order matches ``graph.nodes()``. Falls back to a deterministic spring
    layout if the distance matrix is degenerate (too few nodes, or MDS fails to
    converge on a pathological kernel).
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n < 3:
        return np.array([[float(i), 0.0] for i in range(n)], dtype=float)

    dist = diffusion_distance_matrix(graph, tau, normalized=normalized)
    if not np.isfinite(dist).all() or dist.max() <= 0.0:
        pos = nx.spring_layout(graph, seed=seed, weight="weight")
        return np.array([pos[nd] for nd in nodes], dtype=float)

    try:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            normalized_stress=False,
            n_init=1,
            max_iter=300,
        )
        coords = mds.fit_transform(dist)
    except Exception:  # noqa: BLE001 - degrade to spring layout, never crash a plot
        pos = nx.spring_layout(graph, seed=seed, weight="weight")
        coords = np.array([pos[nd] for nd in nodes], dtype=float)

    # Normalise to a stable [-1, 1] box so the frontend can scale predictably.
    coords = np.asarray(coords, dtype=float)
    span = np.abs(coords).max()
    if span > 0:
        coords = coords / span
    return coords
