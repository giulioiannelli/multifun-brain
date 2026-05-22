"""Partition comparison metrics (Adjusted Rand Index).

Compare cluster assignments produced at different tau scales (or across
runs / contrasts) using the Adjusted Rand Index.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.special import comb

__all__ = ["adjusted_rand_index", "compare_partition_sets"]


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
        Each dict must have ``'tau'`` and ``'partition'`` keys (as
        returned by
        :func:`multifunbrain.analysis.lrg.hierarchical_partitions_from_corr`).

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
