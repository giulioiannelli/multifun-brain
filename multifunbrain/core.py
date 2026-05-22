"""Core helper functions for :mod:`multifunbrain`.

Holds foundational utilities used across the package. The
:func:`band_filter` Butterworth helper has moved to
:mod:`multifunbrain.processing.temporal`; it is re-exported here for
back-compat so ``from multifunbrain.core import band_filter`` keeps working.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Re-exported for back-compat — canonical home is processing.temporal
from .processing.temporal import band_filter

__all__ = ["hello_brain", "band_filter", "marchenko_pastur_density"]


def hello_brain(name: str) -> str:
    """Return a friendly greeting.

    Parameters
    ----------
    name:
        Name or identifier that should appear in the greeting.
    """

    return f"Hello, {name}! Welcome to multifun-brain."


def marchenko_pastur_density(
    eigenvalues: ArrayLike,
    gamma: float,
    sigma: float = 1.0,
) -> np.ndarray:
    r"""Evaluate the Marchenko–Pastur density for given eigenvalues.

    Parameters
    ----------
    eigenvalues:
        Iterable of eigenvalues at which to evaluate the density.
    gamma:
        Aspect ratio of the random matrix (``p / n``). Must be positive.
    sigma:
        Noise variance term. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Density values for each eigenvalue. Values outside the support are zero.

    Notes
    -----
    The Marchenko–Pastur distribution has support :math:`[(\\lambda_-), (\\lambda_+)]`
    where :math:`\\lambda_{\\pm} = \sigma (1 \pm \sqrt{\min(1, \gamma)})^2`.
    """

    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    lam = np.asarray(eigenvalues, dtype=float)

    if gamma <= 1:
        lambda_min = sigma * (1 - np.sqrt(gamma)) ** 2
        lambda_max = sigma * (1 + np.sqrt(gamma)) ** 2
    else:
        inv_gamma = 1 / gamma
        lambda_min = sigma * (1 - np.sqrt(inv_gamma)) ** 2
        lambda_max = sigma * (1 + np.sqrt(inv_gamma)) ** 2

    density = np.zeros_like(lam, dtype=float)
    mask = (lam >= lambda_min) & (lam <= lambda_max)
    lam_valid = lam[mask]
    if lam_valid.size:
        density[mask] = (
            1.0
            / (2 * np.pi * sigma * gamma * lam_valid)
            * np.sqrt((lambda_max - lam_valid) * (lam_valid - lambda_min))
        )
    return density
