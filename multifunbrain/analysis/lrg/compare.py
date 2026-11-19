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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    "HierarchyComparison",
    "compare_hierarchies",
    "null_shift_distribution",
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


@dataclass
class HierarchyComparison:
    """Result of :func:`compare_hierarchies` on a shared leaf (ROI) set.

    All per-ROI arrays are aligned 1-to-1 with ``common`` (the sorted
    intersection of the two trees' atlas indices).
    """

    common: np.ndarray  # atlas indices present in both trees (sorted)
    bakers_gamma: float  # Kendall-tau of cophenetic ranks on the common set
    cophenetic_r: float  # Pearson r of cophenetic distances on the common set
    shift: np.ndarray  # per-ROI cophenetic-rank shift in [0, 1] (observed)
    null_median: np.ndarray  # per-ROI null median (NaN where no null)
    null_p95: np.ndarray  # per-ROI null 95th percentile (NaN where no null)
    null_n_samples: np.ndarray  # per-ROI number of null samples
    calibrated: np.ndarray  # observed - null_median (NaN where no null)
    pooled_null: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def has_null(self) -> bool:
        return bool(np.isfinite(self.null_median).any())

    @property
    def n_common(self) -> int:
        return int(self.common.size)


def _aligned_cophenetic_indices(
    leaf_atlas_a: ArrayLike, leaf_atlas_b: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(common, idx_a, idx_b)`` for the shared atlas indices of two leaf sets."""
    a = np.asarray(leaf_atlas_a, dtype=int)
    b = np.asarray(leaf_atlas_b, dtype=int)
    common = np.array(sorted(set(a.tolist()) & set(b.tolist())), dtype=int)
    pos_a = {int(r): k for k, r in enumerate(a.tolist())}
    pos_b = {int(r): k for k, r in enumerate(b.tolist())}
    idx_a = np.array([pos_a[r] for r in common.tolist()], dtype=int)
    idx_b = np.array([pos_b[r] for r in common.tolist()], dtype=int)
    return common, idx_a, idx_b


def compare_hierarchies(
    linkage_a: np.ndarray,
    leaf_atlas_a: ArrayLike,
    linkage_b: np.ndarray,
    leaf_atlas_b: ArrayLike,
    *,
    null_per_atlas: dict[int, list[float]] | None = None,
) -> HierarchyComparison:
    """Full pairwise dendrogram comparison on the shared leaf (ROI) set.

    The matplotlib-free compute core behind the collaborator's ``fig6`` figure:
    Baker's gamma and cophenetic correlation on the intersection of the two
    trees' leaves, the per-ROI cophenetic-rank shift (0 = same neighbourhood,
    1 = maximally reorganised), and — if a null distribution is supplied — the
    calibrated shift ``observed - null_median`` (> 0 reorganised beyond chance,
    < 0 more stable than chance).

    Parameters
    ----------
    linkage_a, linkage_b : np.ndarray
        SciPy linkage matrices (e.g. from :func:`linkage_at_tau_min`).
    leaf_atlas_a, leaf_atlas_b : array-like of int
        Atlas index of each linkage leaf, in leaf order. The two trees may be
        built on different surviving-ROI sets; they are intersected.
    null_per_atlas : dict, optional
        ``{atlas_index: [shift, ...]}`` from :func:`null_shift_distribution`.

    Returns
    -------
    HierarchyComparison
    """
    common, idx_a, idx_b = _aligned_cophenetic_indices(leaf_atlas_a, leaf_atlas_b)
    n = int(common.size)
    if n < 4:
        empty = np.full(n, np.nan)
        return HierarchyComparison(
            common=common, bakers_gamma=float("nan"), cophenetic_r=float("nan"),
            shift=np.zeros(n), null_median=empty, null_p95=empty,
            null_n_samples=np.zeros(n, dtype=int), calibrated=empty,
        )

    coph_a = squareform(cophenet(linkage_a))[np.ix_(idx_a, idx_a)]
    coph_b = squareform(cophenet(linkage_b))[np.ix_(idx_b, idx_b)]
    iu = np.triu_indices(n, k=1)
    bg, _ = kendalltau(coph_a[iu], coph_b[iu])
    cc, _ = pearsonr(coph_a[iu], coph_b[iu])

    shift = per_leaf_cophenetic_shift(linkage_a, linkage_b, idx_a, idx_b)

    null_median = np.full(n, np.nan, dtype=float)
    null_p95 = np.full(n, np.nan, dtype=float)
    null_n = np.zeros(n, dtype=int)
    pooled: list[float] = []
    if null_per_atlas is not None:
        for i, atlas_idx in enumerate(common.tolist()):
            samples = null_per_atlas.get(int(atlas_idx), [])
            if len(samples) >= 3:
                null_median[i] = float(np.median(samples))
                null_p95[i] = float(np.percentile(samples, 95))
                null_n[i] = len(samples)
            pooled.extend(float(s) for s in samples)
    calibrated = shift - np.where(np.isfinite(null_median), null_median, 0.0)
    calibrated = np.where(np.isfinite(null_median), calibrated, np.nan)

    return HierarchyComparison(
        common=common,
        bakers_gamma=float(bg),
        cophenetic_r=float(cc),
        shift=shift,
        null_median=null_median,
        null_p95=null_p95,
        null_n_samples=null_n,
        calibrated=calibrated,
        pooled_null=np.asarray(pooled, dtype=float),
    )


def null_shift_distribution(
    graph_a,
    label_a: str,
    graph_b,
    label_b: str,
    *,
    n_surrogates: int,
    cache_dir: str | Path,
    rng_seed: int = 0,
    linkage_fn: Callable | None = None,
) -> dict[int, list[float]]:
    """Strength-preserving null distribution of per-ROI cophenetic shifts.

    For each surrogate ``i``, independently strength-preserving-rewire both
    graphs (Rubinov & Sporns 2011), build their LRG linkages, and record the
    per-ROI shift on the common ROIs. Aggregated per atlas index. Surrogate
    linkages are cached on disk (resumable), so re-running a pair is cheap.

    This is the right null for claims about diffusion-on-network structure: it
    asks whether the observed reorganisation exceeds what edge-randomisation at
    fixed node strength would produce.
    """
    from ...processing.null_models import cached_surrogate_linkages
    from .partitions import linkage_at_tau_min

    if linkage_fn is None:
        linkage_fn = linkage_at_tau_min
    cache_dir = Path(cache_dir)
    surrogate_cache = cache_dir / "surrogates"

    # The surrogate-cache filename stem encodes the rng seed, so that comparing a
    # graph against ITSELF (label_a == label_b) still draws two INDEPENDENT
    # surrogate sets (seeds ``rng_seed`` and ``rng_seed + 100_000``) rather than
    # graph B silently reloading graph A's file and collapsing the null to zero.
    seed_a = rng_seed
    seed_b = rng_seed + 100_000
    surrogates_a = cached_surrogate_linkages(
        graph_a, f"{label_a}__s{seed_a}", n_surrogates, surrogate_cache,
        linkage_fn=linkage_fn, rng_seed=seed_a,
    )
    surrogates_b = cached_surrogate_linkages(
        graph_b, f"{label_b}__s{seed_b}", n_surrogates, surrogate_cache,
        linkage_fn=linkage_fn, rng_seed=seed_b,
    )

    null_per_atlas: dict[int, list[float]] = {}
    n_eff = min(len(surrogates_a), len(surrogates_b))
    for i in range(n_eff):
        common, idx_a, idx_b = _aligned_cophenetic_indices(
            surrogates_a[i]["leaf_atlas_idx"], surrogates_b[i]["leaf_atlas_idx"]
        )
        if common.size < 4:
            continue
        shift = per_leaf_cophenetic_shift(
            surrogates_a[i]["linkage"], surrogates_b[i]["linkage"], idx_a, idx_b
        )
        for k, atlas_idx in enumerate(common.tolist()):
            null_per_atlas.setdefault(int(atlas_idx), []).append(float(shift[k]))
    return null_per_atlas


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
