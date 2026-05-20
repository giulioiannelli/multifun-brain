"""Descriptive analysis of raw signed dense correlation networks.

Section 1 of the analysis pipeline: characterise the correlation matrix
*before* any thresholding or absolute-value transformation.  Includes
weight distribution, eigenvalue spectrum with RMT comparison, precision
matrix / partial correlations, signed-Laplacian spectral analysis, and
dense signed-network summary metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kurtosis, skew

from ..core import marchenko_pastur_density

__all__ = [
    "weight_distribution_analysis",
    "correlation_spectrum_analysis",
    "compute_precision_matrix",
    "signed_laplacian_and_spectrum",
    "signed_laplacian_analysis",
    "signed_network_metrics",
    "descriptive_report",
]


# ──────────────────────────────────────────────────────────────────────
# 1. Weight distribution
# ──────────────────────────────────────────────────────────────────────


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

    hist_counts, hist_edges = np.histogram(weights, bins=n_bins) if n_total > 0 else (np.array([]), np.array([]))

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


# ──────────────────────────────────────────────────────────────────────
# 2. Spectral analysis & RMT validation
# ──────────────────────────────────────────────────────────────────────


def correlation_spectrum_analysis(
    corr: np.ndarray,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> dict[str, Any]:
    """Eigenvalue spectrum of a correlation matrix with optional RMT comparison.

    Computes the full eigenvalue spectrum and, when *gamma* is provided,
    compares against the Marchenko-Pastur distribution to separate signal
    eigenvalues from noise.

    Parameters
    ----------
    corr : np.ndarray
        Square correlation matrix.
    gamma : float or None
        Aspect ratio ``p / n`` (number of channels / number of samples).
        If *None*, only the raw spectrum is returned.
    sigma : float
        Noise variance for the MP density.

    Returns
    -------
    dict
        Eigenvalues, eigenvectors, MP bounds, signal/noise partition, etc.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # eigh returns ascending order -- keep that convention
    total_var = np.sum(np.abs(eigenvalues))
    explained = np.abs(eigenvalues) / total_var if total_var > 0 else eigenvalues * 0

    result: dict[str, Any] = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained,
        "largest_eigenvalue": float(eigenvalues[-1]),
    }

    if gamma is not None:
        gamma_eff = min(gamma, 1.0)
        lam_plus = sigma * (1 + np.sqrt(gamma_eff)) ** 2
        lam_minus = sigma * (1 - np.sqrt(gamma_eff)) ** 2

        mp_density = marchenko_pastur_density(eigenvalues, gamma=gamma, sigma=sigma)

        signal_mask = eigenvalues > lam_plus
        noise_mask = (eigenvalues >= lam_minus) & (eigenvalues <= lam_plus)

        result.update(
            {
                "mp_lambda_plus": float(lam_plus),
                "mp_lambda_minus": float(lam_minus),
                "mp_density": mp_density,
                "n_signal": int(np.sum(signal_mask)),
                "n_noise": int(np.sum(noise_mask)),
                "signal_eigenvalues": eigenvalues[signal_mask],
                "noise_eigenvalues": eigenvalues[noise_mask],
            }
        )

    return result


# ──────────────────────────────────────────────────────────────────────
# 3. Precision matrix & partial correlations
# ──────────────────────────────────────────────────────────────────────


def compute_precision_matrix(
    corr: np.ndarray,
    method: str = "direct",
    alpha: float = 0.01,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> dict[str, Any]:
    """Compute the precision matrix and derive partial correlations.

    The precision matrix is the inverse of the covariance / correlation
    matrix.  Its off-diagonal entries, once normalised, yield the partial
    correlation coefficients which measure *direct* statistical
    dependencies between variables.

    Parameters
    ----------
    corr : np.ndarray
        Square correlation matrix.
    method : str
        ``'direct'``
            Moore-Penrose pseudoinverse (works even for singular matrices).
        ``'orie'``
            Optimal Rotationally Invariant Estimator: denoise eigenvalues
            via MP shrinkage, then invert.  Requires *gamma*.
        ``'graphical_lasso'``
            Sparse precision matrix via ``sklearn.covariance.GraphicalLasso``.
    alpha : float
        Regularisation for graphical-lasso.
    gamma : float or None
        Aspect ratio ``p / n`` for the ORIE method.
    sigma : float
        Noise variance for the ORIE method.

    Returns
    -------
    dict
        ``'precision_matrix'``, ``'partial_correlations'``, ``'method'``,
        ``'sparsity'``.
    """
    if method == "direct":
        precision = np.linalg.pinv(corr)

    elif method == "orie":
        if gamma is None:
            raise ValueError("gamma is required for the ORIE method.")
        precision = _orie_precision(corr, gamma, sigma)

    elif method == "graphical_lasso":
        precision = _glasso_precision(corr, alpha)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    partial = _precision_to_partial(precision)
    n = precision.shape[0]
    n_off = n * (n - 1) / 2
    sparsity = float(np.sum(np.abs(partial[np.triu_indices(n, k=1)]) < 1e-8) / n_off) if n_off else 0.0

    return {
        "precision_matrix": precision,
        "partial_correlations": partial,
        "method": method,
        "sparsity": sparsity,
    }


def _orie_precision(
    corr: np.ndarray,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """Optimal Rotationally Invariant Estimator-based precision matrix.

    Denoise the eigenvalue spectrum by shrinking noise eigenvalues (those
    within the MP bulk) towards the bulk mean, then invert.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    gamma_eff = min(gamma, 1.0)
    lam_plus = sigma * (1 + np.sqrt(gamma_eff)) ** 2
    lam_minus = sigma * (1 - np.sqrt(gamma_eff)) ** 2

    denoised = eigenvalues.copy()
    noise_mask = (eigenvalues >= lam_minus) & (eigenvalues <= lam_plus)
    if np.any(noise_mask):
        bulk_mean = float(np.mean(eigenvalues[noise_mask]))
        denoised[noise_mask] = bulk_mean

    # Ensure all eigenvalues are positive for invertibility
    denoised = np.maximum(denoised, 1e-10)

    # Reconstruct denoised covariance and invert
    cov_denoised = (eigenvectors * denoised) @ eigenvectors.T
    return np.linalg.inv(cov_denoised)


def _glasso_precision(corr: np.ndarray, alpha: float) -> np.ndarray:
    """Sparse precision via Graphical Lasso."""
    from sklearn.covariance import GraphicalLasso

    # GraphicalLasso expects a positive-definite matrix; add small ridge
    # to the diagonal to ensure PD if needed.
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals[0] <= 0:
        ridge = abs(eigvals[0]) + 1e-6
        corr_pd = corr + ridge * np.eye(corr.shape[0])
    else:
        corr_pd = corr

    model = GraphicalLasso(alpha=alpha, assume_centered=True, max_iter=500)
    model.fit(corr_pd[np.newaxis, :, :])  # single "sample" = the matrix itself
    return model.precision_


def _precision_to_partial(precision: np.ndarray) -> np.ndarray:
    """Convert a precision matrix to a partial-correlation matrix.

    P_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)  for i != j
    P_ii = 1
    """
    diag = np.diag(precision)
    d = np.sqrt(np.maximum(diag, 0.0))
    d[d == 0] = 1.0  # avoid division by zero
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial


# ──────────────────────────────────────────────────────────────────────
# 4. Signed Laplacian
# ──────────────────────────────────────────────────────────────────────


def signed_laplacian_and_spectrum(
    corr: np.ndarray,
    normalized: bool = False,
) -> tuple:
    """Compute the signed Laplacian ``L = |D| - A`` and its spectrum.

    The signed Laplacian uses the absolute-value degree matrix::

        |D|_ii = sum_j |A_ij|
        L_signed = |D| - A

    For *balanced* signed graphs (no frustrated triangles) this is
    positive semi-definite.  Negative eigenvalues indicate structural
    imbalance / frustration.

    Parameters
    ----------
    corr : np.ndarray
        Signed adjacency matrix (the correlation matrix with zeroed diagonal).
    normalized : bool
        If True, compute the normalised signed Laplacian
        ``L_norm = |D|^{-1/2} L |D|^{-1/2}``.

    Returns
    -------
    L_signed : np.ndarray
        The signed Laplacian matrix.
    eigenvalues : np.ndarray
        Sorted eigenvalues.
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns).
    """
    A = np.array(corr, dtype=float)
    np.fill_diagonal(A, 0.0)

    abs_degree = np.abs(A).sum(axis=1)
    D_abs = np.diag(abs_degree)
    L_signed = D_abs - A

    if normalized:
        d_inv_sqrt = np.zeros_like(abs_degree)
        nonzero = abs_degree > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(abs_degree[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_signed = D_inv_sqrt @ L_signed @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_signed)
    return L_signed, eigenvalues, eigenvectors


def signed_laplacian_analysis(
    corr: np.ndarray,
    normalized: bool = False,
    n_modes: int = 10,
) -> dict[str, Any]:
    """Full signed-Laplacian spectral analysis.

    Parameters
    ----------
    corr : np.ndarray
        Signed adjacency / correlation matrix (diagonal zeroed).
    normalized : bool
        Whether to use the normalised signed Laplacian.
    n_modes : int
        Number of leading eigenmodes to store separately.

    Returns
    -------
    dict
        Signed Laplacian, spectrum, frustration indicators, leading modes.
    """
    L_signed, eigenvalues, eigenvectors = signed_laplacian_and_spectrum(
        corr, normalized=normalized
    )

    neg_mask = eigenvalues < -1e-12
    n_negative = int(np.sum(neg_mask))
    total_abs = np.sum(np.abs(eigenvalues))
    frustration_index = (
        float(np.sum(np.abs(eigenvalues[neg_mask])) / total_abs)
        if total_abs > 0
        else 0.0
    )

    # Spectral gap: difference between the two smallest eigenvalues
    if len(eigenvalues) >= 2:
        spectral_gap = float(eigenvalues[1] - eigenvalues[0])
    else:
        spectral_gap = 0.0

    n_keep = min(n_modes, eigenvectors.shape[1])

    return {
        "L_signed": L_signed,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "n_negative_eigenvalues": n_negative,
        "spectral_gap": spectral_gap,
        "frustration_index": frustration_index,
        "leading_modes": eigenvectors[:, :n_keep],
        "fiedler_vector": eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Dense signed-network metrics
# ──────────────────────────────────────────────────────────────────────


def signed_network_metrics(corr: np.ndarray) -> dict[str, Any]:
    """Compute descriptive metrics for a dense signed weighted network.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix (diagonal should be zeroed).

    Returns
    -------
    dict
        Node-level and global signed-network statistics.
    """
    A = np.array(corr, dtype=float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    n_possible = n * (n - 1) // 2

    # Upper triangle
    idx = np.triu_indices(n, k=1)
    upper = A[idx]

    pos_mask = A > 0
    neg_mask = A < 0

    # Per-node metrics
    strength_pos = (A * pos_mask).sum(axis=1)
    strength_neg = np.abs(A * neg_mask).sum(axis=1)
    strength_total = np.abs(A).sum(axis=1)
    degree_pos = pos_mask.sum(axis=1)
    degree_neg = neg_mask.sum(axis=1)

    total_pos = float(np.sum(upper[upper > 0]))
    total_neg = float(np.sum(np.abs(upper[upper < 0])))
    total_abs = total_pos + total_neg
    balance_ratio = total_pos / total_abs if total_abs > 0 else 0.5

    n_edges = int(np.sum(upper != 0))
    density = n_edges / n_possible if n_possible > 0 else 0.0

    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "n_possible_edges": n_possible,
        "density": density,
        "strength_positive": strength_pos,
        "strength_negative": strength_neg,
        "strength_total": strength_total,
        "degree_positive": degree_pos,
        "degree_negative": degree_neg,
        "mean_strength": float(np.mean(strength_total)),
        "std_strength": float(np.std(strength_total)),
        "balance_ratio": balance_ratio,
        "total_positive_weight": total_pos,
        "total_negative_weight": total_neg,
    }


# ──────────────────────────────────────────────────────────────────────
# 6. Full descriptive report
# ──────────────────────────────────────────────────────────────────────


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
        Square correlation matrix (may still have non-zero diagonal; will
        be zeroed internally where needed).
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
    # Zero diagonal for adjacency-based analyses
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
