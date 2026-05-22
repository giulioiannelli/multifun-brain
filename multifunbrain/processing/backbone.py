"""Backbone extraction: statistically validated edge filtering.

Four methods:

- ``disparity``: Serrano et al. (2009) disparity filter. Best on
  heterogeneous-strength networks; degenerates on dense, near-uniform
  correlation matrices where every node has approximately the same
  strength.
- ``lans``: Locally Adaptive Network Sparsification.
- ``mp_validated``: reconstruct using only eigenvalues outside the
  Marchenko-Pastur bulk.
- ``percolation``: keep edges above the percolation threshold
  ``θ* = min_i max_j w_{ij}`` — the largest cutoff that preserves a
  spanning connected graph. Robust on dense correlation matrices where
  disparity collapses.

All four accept a ``weights`` switch: ``'abs'`` (default) treats the
backbone input as ``|C|`` so anti-correlations contribute as edge
weight; ``'positive'`` clips negatives to zero so the backbone reflects
only co-activation. For fMRI matrices where the negative tail is largely
noise, ``'positive'`` is the principled choice after MP denoising.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from ._giant import matrix_to_graph_giant

__all__ = ["filter_validated"]


def _to_nonneg_weights(corr: np.ndarray, weights: str) -> np.ndarray:
    if weights == "abs":
        return np.abs(corr)
    if weights == "positive":
        return np.clip(corr, 0.0, None)
    raise ValueError(f"Unknown weights mode {weights!r}; use 'abs' or 'positive'.")


def filter_validated(
    corr: np.ndarray,
    method: str = "disparity",
    alpha: float = 0.05,
    gamma: float | None = None,
    sigma: float = 1.0,
    weights: Literal["abs", "positive"] = "abs",
) -> tuple[nx.Graph, list]:
    """Extract statistically significant edges using backbone methods.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix.
    method : str
        ``'disparity'`` | ``'lans'`` | ``'mp_validated'``.
    alpha : float
        Significance level for ``disparity`` and ``lans``.
    gamma : float or None
        Aspect ratio for ``mp_validated`` method.
    sigma : float
        Noise variance for ``mp_validated`` method.
    weights : ``'abs'`` or ``'positive'``
        How to fold the signed matrix into non-negative edge weights before
        filtering. ``'abs'`` keeps anti-correlations as edge weight;
        ``'positive'`` clips negatives to zero. Default is ``'abs'`` for
        back-compatibility; the fMRI pipeline uses ``'positive'``.

    Returns
    -------
    G : nx.Graph
        Filtered unsigned weighted network (giant component).
    removed : list
        Nodes not in the giant component.
    """
    if method == "disparity":
        return _disparity_filter(corr, alpha, weights)
    if method == "lans":
        return _lans_filter(corr, alpha, weights)
    if method == "mp_validated":
        if gamma is None:
            raise ValueError("gamma is required for mp_validated method.")
        return _mp_validated_filter(corr, gamma, sigma, weights)
    if method == "percolation":
        return _percolation_filter(corr, weights)
    raise ValueError(f"Unknown backbone method: {method!r}")


def _percolation_filter(corr: np.ndarray, weights: str) -> tuple[nx.Graph, list]:
    """Keep edges above the percolation threshold ``θ*``.

    ``θ* = min_i max_j w_{ij}`` is the largest threshold at which every
    node still has at least one incident edge. Sparsifies the matrix to a
    spanning weighted graph that loses zero nodes — the natural reference
    backbone for dense correlation matrices.
    """
    A = _to_nonneg_weights(corr, weights).copy()
    np.fill_diagonal(A, 0.0)
    if A.size == 0 or A.max() == 0.0:
        return matrix_to_graph_giant(A)

    max_per_node = A.max(axis=1)
    th_star = float(max_per_node.min())
    filtered = np.where(A >= th_star, A, 0.0)
    return matrix_to_graph_giant(filtered)


def _disparity_filter(corr: np.ndarray, alpha: float, weights: str) -> tuple[nx.Graph, list]:
    """Serrano et al. (2009) disparity filter."""
    A = _to_nonneg_weights(corr, weights).copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    strength = A.sum(axis=1)
    keep = np.zeros_like(A, dtype=bool)

    for i in range(n):
        if strength[i] == 0:
            continue
        neighbours = np.where(A[i] > 0)[0]
        k = len(neighbours)
        if k <= 1:
            keep[i, neighbours] = True
            continue
        for j in neighbours:
            p_ij = A[i, j] / strength[i]
            p_value = (1.0 - p_ij) ** (k - 1)
            if p_value < alpha:
                keep[i, j] = True

    keep = keep | keep.T
    filtered = A * keep
    return matrix_to_graph_giant(filtered)


def _lans_filter(corr: np.ndarray, alpha: float, weights: str) -> tuple[nx.Graph, list]:
    """Locally Adaptive Network Sparsification."""
    A = _to_nonneg_weights(corr, weights).copy()
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    keep = np.zeros_like(A, dtype=bool)

    for i in range(n):
        weights_i = A[i]
        nonzero = weights_i[weights_i > 0]
        if len(nonzero) == 0:
            continue
        quantile = np.quantile(nonzero, 1.0 - alpha)
        keep[i] = weights_i >= quantile

    keep = keep | keep.T
    filtered = A * keep
    return matrix_to_graph_giant(filtered)


def _mp_validated_filter(
    corr: np.ndarray,
    gamma: float,
    sigma: float,
    weights: str,
) -> tuple[nx.Graph, list]:
    """Reconstruct correlation using only signal eigenvalues (outside MP bulk)."""
    C = np.array(corr, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    gamma_eff = min(gamma, 1.0)
    lam_plus = sigma * (1 + np.sqrt(gamma_eff)) ** 2

    signal_mask = eigenvalues > lam_plus
    cleaned_evals = np.where(signal_mask, eigenvalues, 0.0)

    reconstructed = (eigenvectors * cleaned_evals) @ eigenvectors.T
    A = _to_nonneg_weights(reconstructed, weights)
    np.fill_diagonal(A, 0.0)

    return matrix_to_graph_giant(A)
