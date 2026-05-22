"""Characteristic diffusion scales for the Laplacian Renormalisation Group.

Two scales matter on top of the LRG specific-heat curve:

* ``tau_prime = 1 / lambda_max`` — the finest informative resolution, set by
  the spectral radius of ``L`` (zero eigenvalues excluded).
* ``tau_star`` — the location of the long-time specific-heat peak, beyond
  which ``C(tau)`` collapses; the largest scale at which the diffusion still
  resolves community structure.

This module isolates those scale extractors plus a sensible default ``tau``
grid for the per-tau dendrogram pass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["characteristic_scales"]


def characteristic_scales(
    eigenvalues: ArrayLike,
    time_grid: ArrayLike,
    specific_heat: ArrayLike,
    *,
    n_tau: int = 30,
    tau_upper_factor: float = 3.0,
    zero_tol: float = 1e-9,
) -> dict[str, float | np.ndarray]:
    """Extract LRG characteristic scales from a Laplacian spectrum and ``C(tau)``.

    Parameters
    ----------
    eigenvalues : array-like
        Laplacian eigenvalues (any order, any sign of numerical noise).
    time_grid : array-like
        ``tau`` axis used when ``specific_heat`` was computed.
    specific_heat : array-like
        ``C(tau)`` on ``time_grid``.
    n_tau : int
        Number of log-spaced points in the returned default tau grid.
    tau_upper_factor : float
        Default grid extends up to ``tau_upper_factor * tau_star`` to expose
        the post-peak decay.
    zero_tol : float
        Eigenvalues with ``|lambda| < zero_tol`` are treated as numerical
        zeros and excluded from ``lambda_max`` selection.

    Returns
    -------
    dict
        ``{'tau_prime', 'tau_star', 'lambda_max', 'lambda_min_nonzero',
        'tau_grid_default', 'n_zero_eigs'}``.
    """
    evals = np.asarray(eigenvalues, dtype=float)
    t = np.asarray(time_grid, dtype=float)
    C = np.asarray(specific_heat, dtype=float)

    nonzero = evals[np.abs(evals) > zero_tol]
    if nonzero.size == 0:
        raise ValueError("No nonzero eigenvalues — Laplacian is degenerate.")

    lambda_max = float(nonzero.max())
    lambda_min_nonzero = float(nonzero.min())
    tau_prime = 1.0 / lambda_max

    finite = np.isfinite(C)
    if not finite.any():
        raise ValueError("Specific heat is all-NaN; cannot locate tau_star.")
    idx_peak = int(np.argmax(np.where(finite, C, -np.inf)))
    tau_star = float(t[idx_peak])

    tau_lo = tau_prime
    tau_hi = max(tau_star * tau_upper_factor, tau_prime * 10.0)
    tau_grid_default = np.logspace(
        np.log10(tau_lo), np.log10(tau_hi), n_tau
    )

    return {
        "tau_prime": tau_prime,
        "tau_star": tau_star,
        "lambda_max": lambda_max,
        "lambda_min_nonzero": lambda_min_nonzero,
        "tau_grid_default": tau_grid_default,
        "n_zero_eigs": int((np.abs(evals) <= zero_tol).sum()),
    }
