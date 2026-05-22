"""Aggregate descriptive-analysis report (Section 1 of the pipeline)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...processing.partial_correlation import compute_precision_matrix
from .signed import signed_laplacian_analysis, signed_network_metrics
from .spectrum import correlation_spectrum_analysis
from .weights import weight_distribution_analysis

__all__ = ["descriptive_report"]


def descriptive_report(
    corr: np.ndarray,
    gamma: float | None = None,
    sigma: float = 1.0,
    precision_method: str = "direct",
    precision_alpha: float = 0.01,
    n_signed_modes: int = 10,
) -> dict[str, Any]:
    """Run the complete descriptive analysis suite on a raw correlation matrix.

    Orchestrates :func:`weight_distribution_analysis`,
    :func:`correlation_spectrum_analysis`, :func:`compute_precision_matrix`,
    :func:`signed_laplacian_analysis`, and :func:`signed_network_metrics`.

    Parameters
    ----------
    corr : np.ndarray
        Square correlation matrix.
    gamma : float or None
        Aspect ratio for RMT comparison and ORIE precision method.
    sigma : float
        Noise variance for MP density.
    precision_method : str
        Method for :func:`compute_precision_matrix`.
    precision_alpha : float
        Regularisation for graphical-lasso.
    n_signed_modes : int
        Number of signed-Laplacian modes to keep.

    Returns
    -------
    dict
        Nested dict with keys ``'weight_distribution'``, ``'spectrum'``,
        ``'precision'``, ``'signed_laplacian'``, ``'network_metrics'``.
    """
    corr_zd = np.array(corr, dtype=float)
    np.fill_diagonal(corr_zd, 0.0)

    return {
        "weight_distribution": weight_distribution_analysis(corr_zd),
        "spectrum": correlation_spectrum_analysis(corr, gamma=gamma, sigma=sigma),
        "precision": compute_precision_matrix(
            corr,
            method=precision_method,
            alpha=precision_alpha,
            gamma=gamma,
            sigma=sigma,
        ),
        "signed_laplacian": signed_laplacian_analysis(
            corr_zd, normalized=False, n_modes=n_signed_modes
        ),
        "network_metrics": signed_network_metrics(corr_zd),
    }
