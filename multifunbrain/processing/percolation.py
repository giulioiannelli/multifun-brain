"""Percolation-based threshold for correlation networks.

Find the weight threshold at which the first node detaches from the
giant component.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = ["percolation_curve", "percolation_threshold"]


def percolation_curve(
    weights: np.ndarray,
    n_thresholds: int = 40,
    threshold_range: tuple[float, float] | None = None,
    compute_e_inf: bool = True,
) -> dict[str, np.ndarray]:
    """Scan edge-weight thresholds and return ``P_∞(Th)`` and ``E_∞(Th)``.

    For a non-negative weight matrix (typically ``|corr|``), at each
    threshold *Th* keep only edges with ``weight ≥ Th``, build the
    resulting graph, and record:

    - ``P_∞`` = fraction of nodes in the largest connected component
    - ``E_∞`` = fraction of surviving edges at this threshold,
      ``|E(G(Th))| / (N (N-1) / 2)``

    A gradual staircase in ``P_∞(Th)`` is a proxy for functional
    submodules detaching at different correlation strengths; a single
    sharp drop indicates random-graph-like topology.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric non-negative weight matrix.
    n_thresholds : int
        Number of threshold steps (default 40).
    threshold_range : (float, float) or None
        ``(th_min, th_max)``. If ``None``, scan ``[0, |w|.max()]``.
    compute_e_inf : bool
        If ``False``, skip the edge-fraction calculation; the
        returned ``e_inf`` array is filled with ``NaN``.

    Returns
    -------
    dict
        ``{'thresholds': ndarray, 'p_inf': ndarray, 'e_inf': ndarray}``.
    """
    w = np.asarray(weights, dtype=float).copy()
    np.fill_diagonal(w, 0.0)
    n = w.shape[0]

    w_max = float(np.abs(w).max()) if n else 0.0
    if threshold_range is None:
        th_min, th_max = 0.0, w_max
    else:
        th_min, th_max = threshold_range
    thresholds = np.linspace(th_min, th_max, n_thresholds)

    p_inf = np.zeros(n_thresholds)
    e_inf = np.full(n_thresholds, np.nan)
    n_max_edges = n * (n - 1) / 2.0 if n > 1 else 0.0

    if n == 0 or w_max == 0.0:
        if compute_e_inf:
            e_inf[:] = 0.0
        return {"thresholds": thresholds, "p_inf": p_inf, "e_inf": e_inf}

    for i, th in enumerate(thresholds):
        A = w.copy()
        A[A < th] = 0.0
        G = nx.from_numpy_array(A)
        zero_edges = [(u, v) for u, v, d in G.edges(data="weight") if d == 0]
        G.remove_edges_from(zero_edges)

        n_edges = G.number_of_edges()
        if compute_e_inf:
            e_inf[i] = n_edges / n_max_edges if n_max_edges > 0 else 0.0

        if n_edges == 0:
            p_inf[i] = 1.0 / n
            continue

        largest_cc = max(nx.connected_components(G), key=len)
        p_inf[i] = len(largest_cc) / n

    return {"thresholds": thresholds, "p_inf": p_inf, "e_inf": e_inf}


def percolation_threshold(weights: np.ndarray) -> tuple[float, dict]:
    """Find the weight threshold at which the first node detaches.

    For each node, compute its maximum edge weight. The critical
    threshold *Th* is the minimum of those maxima — the point just
    before the weakest-connected node loses its last link.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric non-negative weight matrix (e.g. ``|corr|`` or
        ``max(corr, 0)``). Diagonal is ignored.

    Returns
    -------
    th : float
        Critical threshold. Keeping edges with ``weight >= th``
        guarantees every node retains at least one edge.
    info : dict
        ``'first_detached'`` : int — node index that would detach first.
        ``'p_inf'`` : float — fraction of nodes still connected at *th*.
        ``'e_inf'`` : float — fraction of surviving edges at *th*.
        ``'max_per_node'`` : np.ndarray — strongest edge per node.
    """
    w = weights.copy()
    np.fill_diagonal(w, 0.0)

    max_per_node = w.max(axis=1)
    th = float(max_per_node.min())
    first_detached = int(np.argmin(max_per_node))

    A = w.copy()
    A[A < th] = 0.0
    G = nx.from_numpy_array(A)
    zero_edges = [(u, v) for u, v, d in G.edges(data="weight") if d == 0]
    G.remove_edges_from(zero_edges)

    n = w.shape[0]
    n_max_edges = n * (n - 1) / 2.0 if n > 1 else 0.0
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        p_inf = len(largest_cc) / n
        e_inf = G.number_of_edges() / n_max_edges if n_max_edges > 0 else 0.0
    else:
        p_inf = 0.0
        e_inf = 0.0

    return th, {
        "first_detached": first_detached,
        "p_inf": p_inf,
        "e_inf": e_inf,
        "max_per_node": max_per_node,
    }
