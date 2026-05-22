"""LRG diffusion-based distance for hierarchical clustering."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform

__all__ = ["symmetrized_inverse_distance"]


def symmetrized_inverse_distance(tau, rho_matrix_fn):
    """Compute a symmetric distance matrix from the inverse of the
    diffusion kernel at scale ``tau``.

    Parameters
    ----------
    tau : float
        Diffusion time scale.
    rho_matrix_fn : callable
        Function ``tau -> ndarray`` returning the diffusion kernel.

    Returns
    -------
    dists : ndarray
        Condensed 1D distance array suitable for
        :func:`scipy.cluster.hierarchy.linkage`.
    """
    Trho = 1.0 / rho_matrix_fn(tau)
    Trho = np.maximum(Trho, Trho.T)
    np.fill_diagonal(Trho, 0)
    return squareform(Trho)
