"""Prepare a raw correlation matrix for graph analysis.

Drops dead regions, replaces residual NaN/Inf, symmetrises, clips to
``[-1, 1]``, and optionally zeroes the diagonal.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from .dead_regions import detect_dead_regions

__all__ = ["prepare_correlation_matrix"]

log = logging.getLogger(__name__)


def prepare_correlation_matrix(
    matrix: ArrayLike,
    zero_diagonal: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """Symmetrise and clean a correlation matrix before graph analysis.

    Parameters
    ----------
    matrix : array-like
        Raw correlation matrix (potentially asymmetric or with non-zero diagonal).
    zero_diagonal : bool
        If True, set diagonal to zero.
    clip : bool
        If True, clip values to [-1, 1].

    Returns
    -------
    np.ndarray
        Cleaned, symmetric correlation matrix.

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    corr = np.asarray(matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    dead = detect_dead_regions(corr)
    if len(dead) > 0:
        log.warning(
            "Dropping %d dead region(s) (all-NaN rows/columns): indices %s. "
            "Matrix reduced from %d to %d regions.",
            len(dead),
            dead.tolist(),
            corr.shape[0],
            corr.shape[0] - len(dead),
        )
        keep = np.setdiff1d(np.arange(corr.shape[0]), dead)
        corr = corr[np.ix_(keep, keep)]

    nan_count = int(np.isnan(corr).sum() + np.isinf(corr).sum())
    if nan_count > 0:
        log.warning(
            "After removing dead regions, %d residual NaN/Inf entries "
            "remain; replacing with 0.",
            nan_count,
        )
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    corr = 0.5 * (corr + corr.T)
    if clip:
        corr = np.clip(corr, -1.0, 1.0)
    if zero_diagonal:
        np.fill_diagonal(corr, 0.0)
    return corr
