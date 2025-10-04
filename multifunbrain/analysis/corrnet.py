"""Correlation-network helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..core import marchenko_pastur_density as _marchenko_pastur_density

__all__ = [
    "compute_correlation_matrix",
    "marchenko_pastur",
    "marchenko_pastur_density",
]


def compute_correlation_matrix(timeseries: ArrayLike) -> np.ndarray:
    """Compute the Pearson correlation matrix from multivariate time-series data."""

    ts = np.asarray(timeseries, dtype=float)
    if ts.ndim != 2:
        raise ValueError("timeseries must be a 2D array of shape (n_regions, n_timepoints)")
    return np.corrcoef(ts)


def marchenko_pastur(eigenvalues: ArrayLike, gamma: float) -> np.ndarray:
    """Backwards-compatible wrapper around :func:`marchenko_pastur_density`."""

    return _marchenko_pastur_density(eigenvalues, gamma=gamma, sigma=1.0)


# Public alias so callers can import directly from analysis.corrnet
marchenko_pastur_density = _marchenko_pastur_density
