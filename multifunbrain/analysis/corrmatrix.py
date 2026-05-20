"""Correlation matrix I/O, cleaning, denoising, and LRG clustering utilities.

Functions promoted from ``notebooks/multiscale_correlation_pipeline.ipynb``
so they can be imported, tested, and reused across notebooks and scripts.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import fcluster
from scipy.special import comb

from ..core import marchenko_pastur_density
from .graphutils import compute_normalized_linkage
from .lrglib import (
    compute_optimal_threshold,
    graph_laplacian_and_spectrum,
    rho_matrix,
    symmetrized_inverse_distance,
)

__all__ = [
    "detect_dead_regions",
    "load_correlation_matrix",
    "prepare_correlation_matrix",
    "marchenko_pastur_denoise",
    "hierarchical_partitions_from_corr",
    "adjusted_rand_index",
    "compare_partition_sets",
]


def detect_dead_regions(
    matrix: ArrayLike,
    nan_fraction: float = 1.0,
) -> np.ndarray:
    """Identify dead regions whose off-diagonal entries are (mostly) NaN.

    In fMRI data, a "dead" brain region has zero-variance time series,
    producing an entire row/column of NaN in the correlation matrix.
    These regions must be **dropped** — setting them to zero would
    fabricate correlation data that does not exist.

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

    # Check off-diagonal NaN count per row (ignore diagonal which may be 1 or NaN)
    temp = corr.copy()
    np.fill_diagonal(temp, 0.0)  # make diagonal finite
    nan_per_row = np.isnan(temp).sum(axis=1)
    threshold = nan_fraction * (n - 1)
    dead_mask = nan_per_row >= threshold
    return np.where(dead_mask)[0]


def load_correlation_matrix(path: str | Path) -> np.ndarray:
    """Load a correlation matrix from ``.npy``, ``.npz``, ``.csv``, ``.txt``, or ``.pkl`` files.

    Parameters
    ----------
    path : str or Path
        File path to load.

    Returns
    -------
    np.ndarray
        The loaded correlation matrix.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data[list(data.keys())[0]]
    if path.suffix in {".csv", ".txt"}:
        delimiter = "," if path.suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)
    if path.suffix == ".pkl":
        with open(path, "rb") as fh:
            return np.asarray(pickle.load(fh))

    raise ValueError(f"Unsupported file type: {path.suffix}")


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
    import logging

    log = logging.getLogger(__name__)

    corr = np.asarray(matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    # ── Drop dead regions (rows/columns entirely NaN) ─────────────
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

    # ── Handle remaining sparse NaN / Inf (should be rare) ───────
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


def marchenko_pastur_denoise(
    corr: ArrayLike,
    gamma: float = 0.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Eigenvalue shrinkage denoising using the Marchenko-Pastur support.

    Eigenvalues inside the theoretical MP support are clipped to the nearest
    boundary before reconstructing the matrix.  Adjust *gamma* to reflect
    ``p / n`` where *p* is the number of regions and *n* the number of
    time-point samples.

    Parameters
    ----------
    corr : array-like
        Square correlation matrix.
    gamma : float
        Aspect ratio p/n.
    sigma : float
        Noise variance estimate.

    Returns
    -------
    np.ndarray
        Denoised correlation matrix (symmetrised, diagonal zeroed).
    """
    corr = prepare_correlation_matrix(corr)
    evals, evecs = np.linalg.eigh(corr)
    density = marchenko_pastur_density(evals, gamma=gamma, sigma=sigma)

    lam = np.asarray(evals)
    mask = density > 0
    if not np.any(mask):
        return corr

    lambda_min = lam[mask].min()
    lambda_max = lam[mask].max()
    lam_clipped = np.clip(lam, lambda_min, lambda_max)
    denoised = (evecs * lam_clipped) @ evecs.T
    return prepare_correlation_matrix(denoised)


def hierarchical_partitions_from_corr(
    corr: ArrayLike,
    tau_values: Sequence[float],
    edge_threshold: float = 0.0,
    normalized_laplacian: bool = True,
) -> list[dict]:
    """Run the diffusion / LRG-style clustering pipeline for a correlation matrix.

    Parameters
    ----------
    corr : array-like
        Square correlation matrix.
    tau_values : sequence of float
        Diffusion time scales to evaluate.
    edge_threshold : float
        Minimum edge weight; values below are set to zero.
    normalized_laplacian : bool
        Whether to use the symmetric normalised Laplacian.

    Returns
    -------
    list of dict
        One dict per *tau* value containing keys: ``'tau'``, ``'partition'``,
        ``'linkage_matrix'``, ``'linkage_labels'``, ``'tmax'``,
        ``'flat_threshold'``.
    """
    cleaned = prepare_correlation_matrix(corr)
    if edge_threshold > 0:
        cleaned = np.where(cleaned >= edge_threshold, cleaned, 0.0)

    graph = nx.from_numpy_array(cleaned)
    L, _spectrum = graph_laplacian_and_spectrum(
        graph, weight="weight", normalized=normalized_laplacian
    )

    partitions: list[dict] = []
    for tau in tau_values:
        dists = symmetrized_inverse_distance(tau, lambda t: rho_matrix(t, L))
        linkage_matrix, labels, tmax = compute_normalized_linkage(
            dists, graph, labelList="numbers"
        )
        flat_threshold, _, _, _ = compute_optimal_threshold(linkage_matrix)
        partition = fcluster(linkage_matrix, flat_threshold, criterion="distance")
        partitions.append(
            {
                "tau": float(tau),
                "partition": partition,
                "linkage_matrix": linkage_matrix,
                "linkage_labels": labels,
                "tmax": tmax,
                "flat_threshold": flat_threshold,
            }
        )
    return partitions


def adjusted_rand_index(
    labels_a: Iterable[int],
    labels_b: Iterable[int],
) -> float:
    """Compute the Adjusted Rand Index between two label vectors.

    Parameters
    ----------
    labels_a, labels_b : iterable of int
        Cluster label assignments to compare.

    Returns
    -------
    float
        ARI value in [-1, 1] (1 = perfect agreement).
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("Label arrays must have the same shape.")

    n = labels_a.size
    if n == 0:
        return 0.0

    contingency = np.histogram2d(
        labels_a,
        labels_b,
        bins=(labels_a.max() + 1, labels_b.max() + 1),
    )[0]
    sum_comb_c = comb(contingency.sum(axis=1), 2).sum()
    sum_comb_k = comb(contingency.sum(axis=0), 2).sum()
    sum_comb = comb(contingency, 2).sum()
    total_pairs = comb(n, 2)

    expected_index = (
        (sum_comb_c * sum_comb_k) / total_pairs if total_pairs else 0.0
    )
    max_index = (sum_comb_c + sum_comb_k) / 2
    if max_index == expected_index:
        return 1.0

    return float((sum_comb - expected_index) / (max_index - expected_index))


def compare_partition_sets(
    set_a: list[dict],
    set_b: list[dict],
) -> list[dict]:
    """Compare two sets of partitions across all tau pairs using ARI.

    Parameters
    ----------
    set_a, set_b : list of dict
        Each dict must have ``'tau'`` and ``'partition'`` keys (as returned
        by :func:`hierarchical_partitions_from_corr`).

    Returns
    -------
    list of dict
        Each entry has ``'tau_a'``, ``'tau_b'``, ``'ari'``.
    """
    comparisons: list[dict] = []
    for part_a in set_a:
        for part_b in set_b:
            ari = adjusted_rand_index(part_a["partition"], part_b["partition"])
            comparisons.append(
                {
                    "tau_a": part_a["tau"],
                    "tau_b": part_b["tau"],
                    "ari": ari,
                }
            )
    return comparisons
