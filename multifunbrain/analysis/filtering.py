"""Network filtering / reduction methods for signed correlation matrices.

Section 2 of the analysis pipeline: convert a raw signed dense correlation
matrix into one or more tractable unsigned networks suitable for downstream
LRG multiscale analysis.

Methods included:
- Absolute value + thresholding
- Positive / negative subnetwork separation
- Statistical backbone extraction (disparity filter, LANS, MP-validated)
- Partial-correlation network (via precision matrix)
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .descriptive import compute_precision_matrix
from .graphutils import get_giant_component_leftoff

__all__ = [
    "filter_absolute_threshold",
    "filter_split_sign",
    "filter_validated",
    "filter_partial_correlation",
    "apply_all_filters",
    "percolation_threshold",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _matrix_to_graph_giant(
    matrix: np.ndarray,
) -> tuple[nx.Graph, list]:
    """Build a graph from a non-negative adjacency matrix and extract its
    giant component.

    Parameters
    ----------
    matrix : np.ndarray
        Square non-negative adjacency matrix with zeroed diagonal.

    Returns
    -------
    G : nx.Graph
        Giant connected component.
    removed : list
        Nodes not in the giant component.
    """
    np.fill_diagonal(matrix, 0.0)
    # Remove truly zero entries so the graph is not fully connected
    G_full = nx.from_numpy_array(matrix)
    # Remove zero-weight edges
    zero_edges = [(u, v) for u, v, w in G_full.edges(data="weight") if w == 0]
    G_full.remove_edges_from(zero_edges)

    if G_full.number_of_edges() == 0:
        return G_full, list(G_full.nodes())

    G, removed = get_giant_component_leftoff(G_full, remove_selfloops=True)
    return G, removed


# ──────────────────────────────────────────────────────────────────────
# Percolation-based threshold
# ──────────────────────────────────────────────────────────────────────


def percolation_threshold(
    weights: np.ndarray,
) -> tuple[float, dict]:
    """Find the weight threshold at which the first node detaches.

    For each node, compute its maximum edge weight.  The critical
    threshold *Th* is the minimum of those maxima — the point just
    before the weakest-connected node loses its last link.

    Parameters
    ----------
    weights : np.ndarray
        Symmetric non-negative weight matrix (e.g. ``|corr|`` or
        ``max(corr, 0)``).  Diagonal is ignored.

    Returns
    -------
    th : float
        Critical threshold.  Keeping edges with ``weight >= th``
        guarantees every node retains at least one edge.
    info : dict
        ``'first_detached'`` : int — node index that would detach first.
        ``'p_inf'`` : float — fraction of nodes still connected at *th*.
        ``'e_inf'`` : float — global efficiency of the thresholded graph.
        ``'max_per_node'`` : np.ndarray — strongest edge per node.
    """
    w = weights.copy()
    np.fill_diagonal(w, 0.0)

    max_per_node = w.max(axis=1)
    th = float(max_per_node.min())
    first_detached = int(np.argmin(max_per_node))

    # Build the thresholded graph to compute P_inf and E_inf.
    A = w.copy()
    A[A < th] = 0.0
    G = nx.from_numpy_array(A)
    zero_edges = [(u, v) for u, v, d in G.edges(data="weight") if d == 0]
    G.remove_edges_from(zero_edges)

    n = w.shape[0]
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        p_inf = len(largest_cc) / n
        e_inf = float(nx.global_efficiency(G))
    else:
        p_inf = 0.0
        e_inf = 0.0

    return th, {
        "first_detached": first_detached,
        "p_inf": p_inf,
        "e_inf": e_inf,
        "max_per_node": max_per_node,
    }


# ──────────────────────────────────────────────────────────────────────
# 1. Absolute value + thresholding
# ──────────────────────────────────────────────────────────────────────


def filter_absolute_threshold(
    corr: np.ndarray,
    threshold: float = 0.0,
) -> tuple[nx.Graph, list]:
    """Take absolute value of correlations and threshold.

    Each entry ``|C_ij| >= threshold`` becomes an edge with weight
    ``|C_ij|``.  The giant connected component is extracted.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix.
    threshold : float
        Minimum absolute correlation to retain an edge.

    Returns
    -------
    G : nx.Graph
        Unsigned weighted network from ``|correlation|``.
    removed : list
        Nodes not in the giant component.
    """
    A = np.abs(corr).copy()
    np.fill_diagonal(A, 0.0)
    A[A < threshold] = 0.0
    return _matrix_to_graph_giant(A)


# ──────────────────────────────────────────────────────────────────────
# 2. Positive / negative subnetwork separation
# ──────────────────────────────────────────────────────────────────────


def filter_split_sign(
    corr: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Separate into positive-only and negative-only subnetworks.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix.
    threshold : float
        Minimum weight to keep (applied after sign separation).

    Returns
    -------
    dict
        ``'positive'``: nx.Graph (edges where ``C_ij > threshold``),
        ``'negative'``: nx.Graph (edges where ``C_ij < -threshold``,
        weight = ``|C_ij|``),
        ``'positive_nodes_removed'``, ``'negative_nodes_removed'``.
    """
    C = np.array(corr, dtype=float)
    np.fill_diagonal(C, 0.0)

    # Positive subnetwork
    pos = C.copy()
    pos[pos <= threshold] = 0.0

    # Negative subnetwork (store absolute values as weights)
    neg = np.abs(C.copy())
    neg[C >= -threshold] = 0.0

    G_pos, rem_pos = _matrix_to_graph_giant(pos)
    G_neg, rem_neg = _matrix_to_graph_giant(neg)

    return {
        "positive": G_pos,
        "negative": G_neg,
        "positive_nodes_removed": rem_pos,
        "negative_nodes_removed": rem_neg,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Statistical validation / backbone extraction
# ──────────────────────────────────────────────────────────────────────


def filter_validated(
    corr: np.ndarray,
    method: str = "disparity",
    alpha: float = 0.05,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> tuple[nx.Graph, list]:
    """Extract statistically significant edges using backbone methods.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix.
    method : str
        ``'disparity'``
            Serrano et al. (2009) disparity filter on ``|correlations|``.
        ``'lans'``
            Locally Adaptive Network Sparsification: uses the empirical
            CDF of weights to evaluate significance per node.
        ``'mp_validated'``
            Reconstruct the matrix using only eigenvalues outside the
            Marchenko-Pastur bulk (signal eigenvalues).  Requires *gamma*.
    alpha : float
        Significance level for ``disparity`` and ``lans``.
    gamma : float or None
        Aspect ratio for ``mp_validated`` method.
    sigma : float
        Noise variance for ``mp_validated`` method.

    Returns
    -------
    G : nx.Graph
        Filtered unsigned weighted network (giant component).
    removed : list
        Nodes not in the giant component.
    """
    if method == "disparity":
        return _disparity_filter(corr, alpha)
    if method == "lans":
        return _lans_filter(corr, alpha)
    if method == "mp_validated":
        if gamma is None:
            raise ValueError("gamma is required for mp_validated method.")
        return _mp_validated_filter(corr, gamma, sigma)
    raise ValueError(f"Unknown backbone method: {method!r}")


def _disparity_filter(
    corr: np.ndarray,
    alpha: float,
) -> tuple[nx.Graph, list]:
    """Serrano et al. (2009) disparity filter.

    For each node *i* with degree *k_i*, an edge (i, j) is significant if
    the probability of observing its normalised weight under a uniform
    null is less than *alpha*.  The null CDF is ``1 - (1 - p_ij)^(k_i - 1)``
    where ``p_ij = w_ij / s_i`` and ``s_i`` is node strength.
    """
    A = np.abs(corr).copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    strength = A.sum(axis=1)
    keep = np.zeros_like(A, dtype=bool)

    for i in range(n):
        if strength[i] == 0:
            continue
        neighbours = np.where(A[i] > 0)[0]
        k = len(neighbours)
        if k <= 1:
            keep[i, neighbours] = True
            continue
        for j in neighbours:
            p_ij = A[i, j] / strength[i]
            # CDF of the uniform null for k-1 degrees of freedom
            p_value = 1.0 - (1.0 - p_ij) ** (k - 1)
            if p_value < alpha:
                keep[i, j] = True

    # Symmetrise: keep edge if significant from either end
    keep = keep | keep.T
    filtered = A * keep
    return _matrix_to_graph_giant(filtered)


def _lans_filter(
    corr: np.ndarray,
    alpha: float,
) -> tuple[nx.Graph, list]:
    """Locally Adaptive Network Sparsification.

    For each node *i*, compute the empirical CDF of its edge weights.
    An edge (i, j) is retained if its weight exceeds the ``(1 - alpha)``
    quantile of node *i*'s weight distribution.  No distributional
    assumption is made.
    """
    A = np.abs(corr).copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    keep = np.zeros_like(A, dtype=bool)

    for i in range(n):
        weights_i = A[i]
        nonzero = weights_i[weights_i > 0]
        if len(nonzero) == 0:
            continue
        quantile = np.quantile(nonzero, 1.0 - alpha)
        keep[i] = weights_i >= quantile

    # Symmetrise
    keep = keep | keep.T
    filtered = A * keep
    return _matrix_to_graph_giant(filtered)


def _mp_validated_filter(
    corr: np.ndarray,
    gamma: float,
    sigma: float,
) -> tuple[nx.Graph, list]:
    """Reconstruct correlation using only signal eigenvalues (outside MP bulk).

    Eigenvalues within the Marchenko-Pastur support are set to zero,
    retaining only statistically significant spectral components.  The
    resulting matrix is taken in absolute value for the unsigned network.
    """
    C = np.array(corr, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    gamma_eff = min(gamma, 1.0)
    lam_plus = sigma * (1 + np.sqrt(gamma_eff)) ** 2
    sigma * (1 - np.sqrt(gamma_eff)) ** 2

    # Zero out noise eigenvalues
    signal_mask = eigenvalues > lam_plus
    cleaned_evals = np.where(signal_mask, eigenvalues, 0.0)

    # Reconstruct
    reconstructed = (eigenvectors * cleaned_evals) @ eigenvectors.T
    A = np.abs(reconstructed)
    np.fill_diagonal(A, 0.0)

    return _matrix_to_graph_giant(A)


# ──────────────────────────────────────────────────────────────────────
# 4. Partial-correlation network
# ──────────────────────────────────────────────────────────────────────


def filter_partial_correlation(
    corr: np.ndarray,
    method: str = "direct",
    threshold: float = 0.0,
    alpha: float = 0.01,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> tuple[nx.Graph, list]:
    """Build a network from partial correlations (precision matrix).

    Partial correlations remove indirect effects, typically producing a
    sparser network of direct statistical dependencies.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix.
    method : str
        Method for :func:`~.descriptive.compute_precision_matrix`.
    threshold : float
        Minimum ``|partial correlation|`` to keep as an edge.
    alpha : float
        Regularisation for graphical-lasso.
    gamma : float or None
        Aspect ratio for ORIE method.
    sigma : float
        Noise variance for ORIE method.

    Returns
    -------
    G : nx.Graph
        Network of |partial correlations| (giant component).
    removed : list
        Nodes not in the giant component.
    """
    result = compute_precision_matrix(
        corr, method=method, alpha=alpha, gamma=gamma, sigma=sigma
    )
    partial = result["partial_correlations"]
    A = np.abs(partial)
    np.fill_diagonal(A, 0.0)
    A[A < threshold] = 0.0

    return _matrix_to_graph_giant(A)


# ──────────────────────────────────────────────────────────────────────
# 5. Apply all filters
# ──────────────────────────────────────────────────────────────────────


def apply_all_filters(
    corr: np.ndarray,
    methods: list | None = None,
    threshold: float | None = None,
    alpha: float = 0.05,
    gamma: float | None = None,
    sigma: float = 1.0,
    precision_method: str = "direct",
    precision_alpha: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Apply selected filtering methods and return a dict of results.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix.
    methods : list of str or None
        Which filters to run.  If *None*, runs all available:
        ``['absolute', 'positive', 'negative', 'disparity', 'mp_validated',
        'partial_correlation']``.
    threshold : float or None
        Edge threshold for absolute / split methods.  When *None*
        (default), the percolation threshold at first-node detachment
        is computed automatically for each filter.
    alpha : float
        Significance level for backbone methods.
    gamma : float or None
        Aspect ratio for MP-based methods.
    sigma : float
        Noise variance for MP-based methods.
    precision_method : str
        Precision matrix estimation method.
    precision_alpha : float
        Regularisation for graphical-lasso.

    Returns
    -------
    dict
        Mapping ``filter_name -> {'graph': nx.Graph, 'nodes_removed': list,
        'percolation': dict (when threshold=None)}``.
    """
    available = {
        "absolute",
        "positive",
        "negative",
        "disparity",
        "lans",
        "mp_validated",
        "partial_correlation",
    }
    if methods is None:
        methods = ["absolute", "positive", "negative", "disparity"]
        if gamma is not None:
            methods.append("mp_validated")
        methods.append("partial_correlation")

    results: dict[str, dict[str, Any]] = {}

    # Pre-compute percolation thresholds when threshold is None (auto).
    abs_th = 0.0
    pos_th = 0.0
    perc_abs: dict = {}
    perc_pos: dict = {}

    if threshold is None:
        if any(m in methods for m in ("absolute",)):
            abs_w = np.abs(corr.copy())
            np.fill_diagonal(abs_w, 0.0)
            abs_th, perc_abs = percolation_threshold(abs_w)

        if "positive" in methods:
            pos_w = np.maximum(corr.copy(), 0.0)
            np.fill_diagonal(pos_w, 0.0)
            pos_th, perc_pos = percolation_threshold(pos_w)
    else:
        abs_th = threshold
        pos_th = threshold

    for m in methods:
        if m not in available:
            raise ValueError(f"Unknown filter method: {m!r}. Available: {available}")

        if m == "absolute":
            G, rem = filter_absolute_threshold(corr, abs_th)
            results[m] = {
                "graph": G, "nodes_removed": rem, "percolation": perc_abs,
            }

        elif m == "positive":
            split = filter_split_sign(corr, pos_th)
            results["positive"] = {
                "graph": split["positive"],
                "nodes_removed": split["positive_nodes_removed"],
                "percolation": perc_pos,
            }

        elif m == "negative":
            # Negative filter: no percolation threshold — keep all negative edges.
            split = filter_split_sign(corr, 0.0)
            results["negative"] = {
                "graph": split["negative"],
                "nodes_removed": split["negative_nodes_removed"],
            }

        elif m in ("disparity", "lans"):
            G, rem = filter_validated(corr, method=m, alpha=alpha)
            results[m] = {"graph": G, "nodes_removed": rem}

        elif m == "mp_validated":
            if gamma is None:
                raise ValueError("gamma required for mp_validated filter.")
            G, rem = filter_validated(
                corr, method="mp_validated", gamma=gamma, sigma=sigma
            )
            results[m] = {"graph": G, "nodes_removed": rem}

        elif m == "partial_correlation":
            G, rem = filter_partial_correlation(
                corr,
                method=precision_method,
                threshold=abs_th if threshold is None else threshold,
                alpha=precision_alpha,
                gamma=gamma,
                sigma=sigma,
            )
            results[m] = {"graph": G, "nodes_removed": rem}

    return results
