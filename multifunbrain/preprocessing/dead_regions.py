"""Detect dead (zero-variance) regions in a correlation matrix.

A dead brain region (zero-variance fMRI time series) produces an entire
row/column of NaN in the correlation matrix. These regions must be
**dropped** — setting them to zero would fabricate correlation data
that does not exist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["detect_dead_regions"]


def detect_dead_regions(
    matrix: ArrayLike,
    nan_fraction: float = 1.0,
) -> np.ndarray:
    """Identify dead regions whose off-diagonal entries are (mostly) NaN.

    Parameters
    ----------
    matrix : array-like
        Square correlation matrix (may contain NaN).
    nan_fraction : float
        Fraction of off-diagonal entries that must be NaN for a region
        to be considered dead.  Default ``1.0`` means the *entire*
        off-diagonal row must be NaN.  Use e.g. ``0.9`` to also catch
        nearly-dead regions.

    Returns
    -------
    np.ndarray
        Integer array of indices of dead regions (may be empty).
    """
    corr = np.asarray(matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return np.array([], dtype=int)
    n = corr.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    temp = corr.copy()
    np.fill_diagonal(temp, 0.0)
    nan_per_row = np.isnan(temp).sum(axis=1)
    threshold = nan_fraction * (n - 1)
    dead_mask = nan_per_row >= threshold
    return np.where(dead_mask)[0]
