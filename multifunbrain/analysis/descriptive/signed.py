"""Signed-network descriptors: signed Laplacian and dense signed metrics.

The signed Laplacian ``L = |D| - A`` is positive semi-definite for
balanced signed graphs (no frustrated triangles). Negative eigenvalues
indicate structural imbalance / frustration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "signed_laplacian_and_spectrum",
    "signed_laplacian_analysis",
    "signed_network_metrics",
]


def signed_laplacian_and_spectrum(
    corr: np.ndarray,
    normalized: bool = False,
) -> tuple:
    """Compute the signed Laplacian ``L = |D| - A`` and its spectrum.

    Parameters
    ----------
    corr : np.ndarray
        Signed adjacency matrix (correlation matrix with zeroed diagonal).
    normalized : bool
        If True, compute the normalised signed Laplacian
        ``L_norm = |D|^{-1/2} L |D|^{-1/2}``.

    Returns
    -------
    L_signed : np.ndarray
        The signed Laplacian matrix.
    eigenvalues : np.ndarray
        Sorted eigenvalues.
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns).
    """
    A = np.array(corr, dtype=float)
    np.fill_diagonal(A, 0.0)

    abs_degree = np.abs(A).sum(axis=1)
    D_abs = np.diag(abs_degree)
    L_signed = D_abs - A

    if normalized:
        d_inv_sqrt = np.zeros_like(abs_degree)
        nonzero = abs_degree > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(abs_degree[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_signed = D_inv_sqrt @ L_signed @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_signed)
    return L_signed, eigenvalues, eigenvectors


def signed_laplacian_analysis(
    corr: np.ndarray,
    normalized: bool = False,
    n_modes: int = 10,
) -> dict[str, Any]:
    """Full signed-Laplacian spectral analysis.

    Parameters
    ----------
    corr : np.ndarray
        Signed adjacency / correlation matrix (diagonal zeroed).
    normalized : bool
        Whether to use the normalised signed Laplacian.
    n_modes : int
        Number of leading eigenmodes to store separately.

    Returns
    -------
    dict
        Signed Laplacian, spectrum, frustration indicators, leading modes.
    """
    L_signed, eigenvalues, eigenvectors = signed_laplacian_and_spectrum(
        corr, normalized=normalized
    )

    neg_mask = eigenvalues < -1e-12
    n_negative = int(np.sum(neg_mask))
    total_abs = np.sum(np.abs(eigenvalues))
    frustration_index = (
        float(np.sum(np.abs(eigenvalues[neg_mask])) / total_abs)
        if total_abs > 0
        else 0.0
    )

    if len(eigenvalues) >= 2:
        spectral_gap = float(eigenvalues[1] - eigenvalues[0])
    else:
        spectral_gap = 0.0

    n_keep = min(n_modes, eigenvectors.shape[1])

    return {
        "L_signed": L_signed,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "n_negative_eigenvalues": n_negative,
        "spectral_gap": spectral_gap,
        "frustration_index": frustration_index,
        "leading_modes": eigenvectors[:, :n_keep],
        "fiedler_vector": eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None,
    }


def signed_network_metrics(corr: np.ndarray) -> dict[str, Any]:
    """Compute descriptive metrics for a dense signed weighted network.

    Parameters
    ----------
    corr : np.ndarray
        Signed correlation matrix (diagonal should be zeroed).

    Returns
    -------
    dict
        Node-level and global signed-network statistics.
    """
    A = np.array(corr, dtype=float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    n_possible = n * (n - 1) // 2

    idx = np.triu_indices(n, k=1)
    upper = A[idx]

    pos_mask = A > 0
    neg_mask = A < 0

    strength_pos = (A * pos_mask).sum(axis=1)
    strength_neg = np.abs(A * neg_mask).sum(axis=1)
    strength_total = np.abs(A).sum(axis=1)
    degree_pos = pos_mask.sum(axis=1)
    degree_neg = neg_mask.sum(axis=1)

    total_pos = float(np.sum(upper[upper > 0]))
    total_neg = float(np.sum(np.abs(upper[upper < 0])))
    total_abs = total_pos + total_neg
    balance_ratio = total_pos / total_abs if total_abs > 0 else 0.5

    n_edges = int(np.sum(upper != 0))
    density = n_edges / n_possible if n_possible > 0 else 0.0

    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "n_possible_edges": n_possible,
        "density": density,
        "strength_positive": strength_pos,
        "strength_negative": strength_neg,
        "strength_total": strength_total,
        "degree_positive": degree_pos,
        "degree_negative": degree_neg,
        "mean_strength": float(np.mean(strength_total)),
        "std_strength": float(np.std(strength_total)),
        "balance_ratio": balance_ratio,
        "total_positive_weight": total_pos,
        "total_negative_weight": total_neg,
    }
