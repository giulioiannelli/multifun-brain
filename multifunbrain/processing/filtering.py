"""Network filtering: convert a signed correlation matrix into unsigned
weighted networks.

Methods:

- :func:`filter_absolute_threshold` — take ``|corr|`` and threshold.
- :func:`filter_split_sign` — separate into positive and negative subnetworks.
- :func:`filter_partial_correlation` — build network from partial correlations.
- :func:`apply_all_filters` — apply selected filters and return a dict.

Backbone-based filters live in :mod:`multifunbrain.processing.backbone`.
Percolation threshold lives in :mod:`multifunbrain.processing.percolation`.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from ._giant import matrix_to_graph_giant
from .backbone import filter_validated
from .partial_correlation import compute_precision_matrix
from .percolation import percolation_threshold

__all__ = [
    "filter_absolute_threshold",
    "filter_split_sign",
    "filter_partial_correlation",
    "apply_all_filters",
]


def filter_absolute_threshold(
    corr: np.ndarray,
    threshold: float = 0.0,
) -> tuple[nx.Graph, list]:
    """Take absolute value of correlations and threshold.

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
    return matrix_to_graph_giant(A)


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
        ``'positive'``, ``'negative'`` (both ``nx.Graph``);
        ``'positive_nodes_removed'``, ``'negative_nodes_removed'``.
    """
    C = np.array(corr, dtype=float)
    np.fill_diagonal(C, 0.0)

    pos = C.copy()
    pos[pos <= threshold] = 0.0

    neg = np.abs(C.copy())
    neg[C >= -threshold] = 0.0

    G_pos, rem_pos = matrix_to_graph_giant(pos)
    G_neg, rem_neg = matrix_to_graph_giant(neg)

    return {
        "positive": G_pos,
        "negative": G_neg,
        "positive_nodes_removed": rem_pos,
        "negative_nodes_removed": rem_neg,
    }


def filter_partial_correlation(
    corr: np.ndarray,
    method: str = "direct",
    threshold: float = 0.0,
    alpha: float = 0.01,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> tuple[nx.Graph, list]:
    """Build a network from partial correlations (precision matrix).

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix.
    method : str
        Method for :func:`compute_precision_matrix`.
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
    return matrix_to_graph_giant(A)


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
        Which filters to run. Default: ``['absolute', 'positive', 'negative',
        'disparity']`` (plus ``'mp_validated'`` if *gamma* is set, plus
        ``'partial_correlation'``).
    threshold : float or None
        Edge threshold for absolute / split methods. When *None* (default),
        the percolation threshold at first-node detachment is computed
        automatically per filter.
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
            results[m] = {"graph": G, "nodes_removed": rem, "percolation": perc_abs}

        elif m == "positive":
            split = filter_split_sign(corr, pos_th)
            results["positive"] = {
                "graph": split["positive"],
                "nodes_removed": split["positive_nodes_removed"],
                "percolation": perc_pos,
            }

        elif m == "negative":
            split = filter_split_sign(corr, 0.0)
            results["negative"] = {
                "graph": split["negative"],
                "nodes_removed": split["negative_nodes_removed"],
            }

        elif m in ("disparity", "lans"):
            G, rem = filter_validated(
                corr, method=m, alpha=alpha, weights="positive"
            )
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
