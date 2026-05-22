"""Multiscale-aware partition and hierarchy comparison.

Four complementary primitives — point-wise (RMI), full-hierarchy
(cophenetic correlation, Baker's gamma), and partition-free LRG
(specific-heat comparison) — replace ARI as the headline metric for
comparing LRG outputs between two networks (e.g. CO2 vs rest).

References
----------
Newman, Cantwell, Young, *Phys. Rev. E* 101, 042304 (2020) — RMI.
Sokal & Rohlf 1962 — cophenetic correlation.
Baker, *J. Am. Stat. Assoc.* 69, 440 (1974) — Baker's gamma.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, pearsonr, rankdata
from sklearn.metrics import mutual_info_score

__all__ = [
    "reduced_mutual_information",
    "cophenetic_correlation",
    "bakers_gamma",
    "per_leaf_cophenetic_shift",
    "specific_heat_comparison",
]


def reduced_mutual_information(
    labels_a: ArrayLike,
    labels_b: ArrayLike,
) -> float:
    """Reduced Mutual Information (Newman, Cantwell, Young 2020).

    The raw mutual information ``I(A;B)`` is biased upward when ``A`` and
    ``B`` have different cluster counts. The reduced form subtracts an
    asymptotic correction for the entropy of contingency tables compatible
    with the row / column sums:

    .. math::

        \\text{RMI}(A, B) \\approx I(A; B) - \\frac{(R_A - 1)(R_B - 1)}{2 N}
                                              \\, \\ln N

    where ``R_A``, ``R_B`` are the number of clusters in ``A`` and ``B``
    and ``N`` is the number of nodes. The correction handles the
    LRG-typical case of partitions with very different optimal cluster
    counts across ``tau``.

    Parameters
    ----------
    labels_a, labels_b : array-like
        Integer label arrays of the same length.

    Returns
    -------
    float
        Reduced mutual information in nats. Equal to ``H(A)`` when both
        partitions agree; near ``0`` for independent partitions; can be
        slightly negative for small ``N`` where the asymptotic correction
        overshoots.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")
    n = a.size
    if n < 2:
        return 0.0

    r_a = int(np.unique(a).size)
    r_b = int(np.unique(b).size)

    mi = float(mutual_info_score(a, b))
    beta = (r_a - 1) * (r_b - 1) * np.log(n) / (2.0 * n)
    return mi - beta


def cophenetic_correlation(
    linkage_a: np.ndarray,
    linkage_b: np.ndarray,
) -> float:
    """Pearson correlation between cophenetic-distance vectors.

    Compares the **full hierarchies** rather than a single cut. Both
    linkage matrices must be on the same node ordering.

    Returns
    -------
    float
        Pearson r between the two condensed cophenetic-distance vectors,
        in ``[-1, 1]``. ``1`` means the two dendrograms induce identical
        ultrametric distances; ``0`` means structurally uncorrelated.
    """
    coph_a = cophenet(linkage_a)
    coph_b = cophenet(linkage_b)
    if coph_a.shape != coph_b.shape:
        raise ValueError(
            f"linkage matrices imply different leaf counts: "
            f"{coph_a.shape} vs {coph_b.shape}"
        )
    if coph_a.std() == 0 or coph_b.std() == 0:
        return float("nan")
    r, _ = pearsonr(coph_a, coph_b)
    return float(r)


def bakers_gamma(
    linkage_a: np.ndarray,
    linkage_b: np.ndarray,
) -> float:
    """Baker's gamma — Kendall-τ between cophenetic rank orders.

    More sensitive to local-topology differences than Pearson cophenetic
    correlation. ``+1`` ⇒ identical rank order of joining heights;
    ``0`` ⇒ no rank agreement; ``-1`` ⇒ reversed ordering.
    """
    coph_a = cophenet(linkage_a)
    coph_b = cophenet(linkage_b)
    if coph_a.shape != coph_b.shape:
        raise ValueError(
            f"linkage matrices imply different leaf counts: "
            f"{coph_a.shape} vs {coph_b.shape}"
        )
    tau, _ = kendalltau(coph_a, coph_b)
    return float(tau)


def per_leaf_cophenetic_shift(
    linkage_a: np.ndarray,
    linkage_b: np.ndarray,
    leaf_indices_a: ArrayLike | None = None,
    leaf_indices_b: ArrayLike | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Per-leaf reorganisation score between two dendrograms.

    For each conceptual leaf (e.g. atlas ROI), compares its full row in
    the cophenetic-distance matrix between ``A`` and ``B`` after
    rank-transforming the two matrices independently:

    .. math::

        d_i = \\frac{1}{(n-1) \\, n_\\text{pairs}}
              \\sum_{j \\ne i}
              |\\operatorname{rank}_A(C_A[i,j])
             - \\operatorname{rank}_B(C_B[i,j])|

    Ranks are taken over the off-diagonal pairs of each matrix, so the
    comparison is invariant to monotonic rescaling of merge heights. A
    high ``d_i`` means leaf ``i`` sits in a structurally different
    position in the two trees.

    The two linkages may use **different leaf orderings** (e.g. each one
    was built on its own set of surviving ROIs after dead-region
    pruning). Use ``leaf_indices_a`` / ``leaf_indices_b`` to align them
    on a shared concept (e.g. the intersection of surviving atlas
    indices).

    Parameters
    ----------
    linkage_a, linkage_b : np.ndarray
        Linkage matrices.
    leaf_indices_a, leaf_indices_b : array-like of int, optional
        Indices into the respective linkage's leaf ordering. Must have
        the same length, and entry ``k`` in each array must refer to
        the **same** conceptual leaf. If both are ``None`` (the default),
        the two linkages are assumed to share an identical leaf
        ordering.
    normalize : bool, default True
        If ``True``, rank values are divided by the number of off-diagonal
        pairs so that ``d_i`` falls in ``[0, 1]``. If ``False``, raw
        rank-difference sums are returned.

    Returns
    -------
    np.ndarray
        Per-leaf shift score in the order given by ``leaf_indices_a``
        (or ``0..n-1`` if not provided).
    """
    coph_a = squareform(cophenet(linkage_a))
    coph_b = squareform(cophenet(linkage_b))

    if leaf_indices_a is None and leaf_indices_b is None:
        if coph_a.shape != coph_b.shape:
            raise ValueError(
                f"linkage matrices imply different leaf counts "
                f"({coph_a.shape} vs {coph_b.shape}); provide "
                f"leaf_indices_a/leaf_indices_b to align them."
            )
    else:
        if leaf_indices_a is None or leaf_indices_b is None:
            raise ValueError(
                "leaf_indices_a and leaf_indices_b must both be given "
                "or both be None."
            )
        idx_a = np.asarray(leaf_indices_a, dtype=int)
        idx_b = np.asarray(leaf_indices_b, dtype=int)
        if idx_a.shape != idx_b.shape:
            raise ValueError(
                f"leaf_indices_a and leaf_indices_b must have the same "
                f"length, got {idx_a.shape} vs {idx_b.shape}."
            )
        coph_a = coph_a[np.ix_(idx_a, idx_a)]
        coph_b = coph_b[np.ix_(idx_b, idx_b)]

    n = coph_a.shape[0]
    if n < 3:
        return np.zeros(n, dtype=float)

    iu = np.triu_indices(n, k=1)
    rank_a = np.zeros_like(coph_a)
    rank_b = np.zeros_like(coph_b)
    rank_a[iu] = rankdata(coph_a[iu])
    rank_b[iu] = rankdata(coph_b[iu])
    rank_a = rank_a + rank_a.T
    rank_b = rank_b + rank_b.T

    n_pairs = n * (n - 1) // 2
    diff = np.abs(rank_a - rank_b)
    np.fill_diagonal(diff, 0.0)
    per_leaf = diff.sum(axis=1) / (n - 1)
    if normalize:
        per_leaf = per_leaf / n_pairs
    return per_leaf


def specific_heat_comparison(
    specific_heat_a: ArrayLike,
    specific_heat_b: ArrayLike,
    time_grid: ArrayLike,
) -> dict[str, float]:
    """Partition-free LRG-native contrast summary from ``C(tau)``.

    Computes peak positions, their shift, and the integrated absolute
    difference along ``log tau``.

    Returns
    -------
    dict
        ``{'tau_star_a', 'tau_star_b', 'delta_tau_star',
        'log_delta_tau_star', 'area_abs_delta'}``.
    """
    c_a = np.asarray(specific_heat_a, dtype=float)
    c_b = np.asarray(specific_heat_b, dtype=float)
    t = np.asarray(time_grid, dtype=float)
    if c_a.shape != c_b.shape or c_a.shape != t.shape:
        raise ValueError("All inputs must have the same length")

    finite_a = np.isfinite(c_a)
    finite_b = np.isfinite(c_b)
    if not finite_a.any() or not finite_b.any():
        return {
            "tau_star_a": float("nan"),
            "tau_star_b": float("nan"),
            "delta_tau_star": float("nan"),
            "log_delta_tau_star": float("nan"),
            "area_abs_delta": float("nan"),
        }

    tau_star_a = float(t[int(np.argmax(np.where(finite_a, c_a, -np.inf)))])
    tau_star_b = float(t[int(np.argmax(np.where(finite_b, c_b, -np.inf)))])

    log_t = np.log10(t)
    delta = np.abs(c_a - c_b)
    area = float(np.trapezoid(delta, log_t))

    return {
        "tau_star_a": tau_star_a,
        "tau_star_b": tau_star_b,
        "delta_tau_star": tau_star_a - tau_star_b,
        "log_delta_tau_star": np.log10(tau_star_a) - np.log10(tau_star_b),
        "area_abs_delta": area,
    }
