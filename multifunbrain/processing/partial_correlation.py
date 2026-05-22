"""Precision matrix and partial-correlation estimation.

The precision matrix is the inverse of the covariance/correlation
matrix. Its off-diagonal entries, once normalised, yield the partial
correlation coefficients which measure *direct* statistical
dependencies between variables.

Three estimation methods are supported:

- ``direct``: Moore-Penrose pseudoinverse (works even for singular matrices).
- ``orie``: Optimal Rotationally Invariant Estimator (MP-shrunken
  eigenvalues, then invert). Requires ``gamma``.
- ``graphical_lasso``: Sparse precision via
  :class:`sklearn.covariance.GraphicalLasso`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["compute_precision_matrix"]


def compute_precision_matrix(
    corr: np.ndarray,
    method: str = "direct",
    alpha: float = 0.01,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> dict[str, Any]:
    """Compute the precision matrix and derive partial correlations.

    Parameters
    ----------
    corr : np.ndarray
        Square correlation matrix.
    method : str
        ``'direct'`` | ``'orie'`` | ``'graphical_lasso'``.
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


def _orie_precision(corr: np.ndarray, gamma: float, sigma: float) -> np.ndarray:
    """Optimal Rotationally Invariant Estimator-based precision matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    gamma_eff = min(gamma, 1.0)
    lam_plus = sigma * (1 + np.sqrt(gamma_eff)) ** 2
    lam_minus = sigma * (1 - np.sqrt(gamma_eff)) ** 2

    denoised = eigenvalues.copy()
    noise_mask = (eigenvalues >= lam_minus) & (eigenvalues <= lam_plus)
    if np.any(noise_mask):
        bulk_mean = float(np.mean(eigenvalues[noise_mask]))
        denoised[noise_mask] = bulk_mean

    denoised = np.maximum(denoised, 1e-10)
    cov_denoised = (eigenvectors * denoised) @ eigenvectors.T
    return np.linalg.inv(cov_denoised)


def _glasso_precision(corr: np.ndarray, alpha: float) -> np.ndarray:
    """Sparse precision via Graphical Lasso."""
    from sklearn.covariance import GraphicalLasso

    eigvals = np.linalg.eigvalsh(corr)
    if eigvals[0] <= 0:
        ridge = abs(eigvals[0]) + 1e-6
        corr_pd = corr + ridge * np.eye(corr.shape[0])
    else:
        corr_pd = corr

    model = GraphicalLasso(alpha=alpha, assume_centered=True, max_iter=500)
    model.fit(corr_pd[np.newaxis, :, :])
    return model.precision_


def _precision_to_partial(precision: np.ndarray) -> np.ndarray:
    """Convert a precision matrix to a partial-correlation matrix.

    P_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)  for i != j
    P_ii = 1
    """
    diag = np.diag(precision)
    d = np.sqrt(np.maximum(diag, 0.0))
    d[d == 0] = 1.0
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial
