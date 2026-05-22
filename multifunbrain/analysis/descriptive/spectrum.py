"""Eigenvalue spectrum analysis with optional Marchenko-Pastur comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import marchenko_pastur_density

__all__ = ["correlation_spectrum_analysis"]


def correlation_spectrum_analysis(
    corr: np.ndarray,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> dict[str, Any]:
    """Eigenvalue spectrum of a correlation matrix with optional RMT comparison.

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
