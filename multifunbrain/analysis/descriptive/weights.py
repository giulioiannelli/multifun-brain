"""Weight distribution analysis of a signed correlation matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kurtosis, skew

__all__ = ["weight_distribution_analysis"]


def weight_distribution_analysis(
    corr: np.ndarray,
    n_bins: int = 100,
) -> dict[str, Any]:
    """Analyse the weight distribution of a signed correlation matrix.

    Only the upper triangle (excluding diagonal) is considered so that
    each edge is counted once.

    Parameters
    ----------
    corr : np.ndarray
        Square (N, N) correlation matrix.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Descriptive statistics and histogram of edge weights.
    """
    idx = np.triu_indices_from(corr, k=1)
    weights = corr[idx]
    weights = weights[np.isfinite(weights)]

    pos = weights[weights > 0]
    neg = weights[weights < 0]
    n_total = len(weights)

    hist_counts, hist_edges = (
        np.histogram(weights, bins=n_bins) if n_total > 0 else (np.array([]), np.array([]))
    )

    return {
        "all_weights": weights,
        "positive_weights": pos,
        "negative_weights": neg,
        "n_positive": len(pos),
        "n_negative": len(neg),
        "n_zero": int(np.sum(weights == 0)),
        "n_total": n_total,
        "frac_positive": len(pos) / n_total if n_total else 0.0,
        "frac_negative": len(neg) / n_total if n_total else 0.0,
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "skewness": float(skew(weights)),
        "kurtosis": float(kurtosis(weights)),
        "mean_positive": float(np.mean(pos)) if len(pos) else 0.0,
        "mean_negative": float(np.mean(neg)) if len(neg) else 0.0,
        "median": float(np.median(weights)),
        "histogram": (hist_counts, hist_edges),
    }
