"""Marchenko-Pastur denoising of correlation matrices.

Eigenvalues inside the theoretical MP support are clipped to the nearest
boundary before reconstructing the matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..core import marchenko_pastur_density
from .prepare import prepare_correlation_matrix

__all__ = ["marchenko_pastur_denoise", "marchenko_pastur_density"]


def marchenko_pastur_denoise(
    corr: ArrayLike,
    gamma: float = 0.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Eigenvalue shrinkage denoising using the Marchenko-Pastur support.

    Reconstructs the correlation matrix after replacing every eigenvalue
    that lies inside the MP bulk with the upper bulk boundary
    ``lambda_+ = sigma * (1 + sqrt(gamma))^2``, while keeping signal
    eigenvalues (``lambda > lambda_+``) intact. This is the canonical
    Laloux–Cizeau–Bouchaud–Potters (1999) cleaning, simplified to a flat
    plateau for the bulk so the noise floor is uniform.

    The eigendecomposition is performed on the **unit-diagonal** form
    (a true correlation matrix has trace ``N``); the routine restores a
    unit diagonal if the caller passed a zero-diagonal adjacency, and the
    diagonal is zeroed again at the end via ``prepare_correlation_matrix``.

    Parameters
    ----------
    corr : array-like
        Square correlation matrix.
    gamma : float
        Aspect ratio ``p / n`` (regions / time-point samples).
    sigma : float
        Noise variance estimate.

    Returns
    -------
    np.ndarray
        Denoised correlation matrix (symmetrised, diagonal zeroed).
    """
    corr_input = np.asarray(corr, dtype=float)
    if corr_input.ndim != 2 or corr_input.shape[0] != corr_input.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    corr_for_eig = 0.5 * (corr_input + corr_input.T)
    if not np.all(np.isfinite(corr_for_eig)):
        corr_for_eig = np.nan_to_num(corr_for_eig, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(np.diag(corr_for_eig), 0.0):
        np.fill_diagonal(corr_for_eig, 1.0)

    evals, evecs = np.linalg.eigh(corr_for_eig)
    gamma_eff = float(min(max(gamma, 1e-9), 1.0))
    lambda_plus = sigma * (1.0 + np.sqrt(gamma_eff)) ** 2

    in_bulk = evals <= lambda_plus
    if not np.any(in_bulk):
        return prepare_correlation_matrix(corr_for_eig)

    lam_clipped = np.where(in_bulk, lambda_plus, evals)
    denoised = (evecs * lam_clipped) @ evecs.T
    return prepare_correlation_matrix(denoised)
