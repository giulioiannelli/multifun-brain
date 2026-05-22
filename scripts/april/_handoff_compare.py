"""Pairwise dendrogram comparison for the collaborator handoff.

For one pair of correlation matrices, this module:

1. Builds the LANS-backbone for each (via ``chain_pipeline``) and the LRG
   diffusion-distance linkage at τ_min = 1/λ_max (via ``linkage_at_tau_min``).
2. Computes the per-leaf cophenetic-rank shift on the common-ROI
   sub-trees (Spearman-like per-ROI reorganisation score in [0, 1]).
3. Builds a **strength-preserving null** by rewiring each LANS backbone
   while keeping each node's strength approximately constant
   (Rubinov & Sporns 2011 — see :mod:`multifunbrain.processing.null_models`).
   Surrogate dendrograms are cached so subsequent comparisons reuse them.
4. Emits the figure + a per-ROI CSV table.

Three null choices are exposed:

- ``strength-preserving`` (default): test whether the observed shift
  exceeds what edges-randomised-at-fixed-strength would produce. Right
  null for claims about diffusion-on-network structure.
- ``subject-split``: within-condition balanced split-half over the 6
  per-subject TS, the proper sampling-noise null. Kept for sanity
  checks at the global level where per-subject TS exist.
- ``none``: show raw observed shift only.

The figure (``fig3_compare.pdf``) and table (``fig3_per_roi_shift.csv``)
match the layout already validated in the previous iteration of this
script (three brain orientation rows + histogram + bar chart).
"""

from __future__ import annotations

import logging
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from nilearn import plotting

from _fig_common import (
    TS_BASE,
    chain_pipeline,
    clean_atlas_label,
    linkage_at_tau_min,
)
from multifunbrain.analysis.lrg.compare import per_leaf_cophenetic_shift
from multifunbrain.processing.null_models import cached_surrogate_linkages

log = logging.getLogger(__name__)

__all__ = [
    "build_fig3_compare",
    "compute_pair",
    "load_subject_ts",
    "strength_preserving_null",
    "subject_split_null",
]


# ────────────────────────────────────────────────────────────────────
# Colormap helpers
# ────────────────────────────────────────────────────────────────────


def _cmap_transparent_at_zero(name: str = "magma_r", n: int = 256) -> ListedColormap:
    base = mpl.colormaps[name](np.linspace(0.0, 1.0, n)).copy()
    base[0] = [0.0, 0.0, 0.0, 0.0]
    return ListedColormap(base, name=f"{name}_t0")


def _cmap_transparent_at_center(name: str = "RdBu_r", n: int = 256) -> ListedColormap:
    base = mpl.colormaps[name](np.linspace(0.0, 1.0, n)).copy()
    base[n // 2, 3] = 0.0
    return ListedColormap(base, name=f"{name}_tc")


def _shift_to_roi_image(
    score_per_roi: dict[int, float], atlas_data: np.ndarray, atlas_img
):
    out = np.full_like(atlas_data, fill_value=np.nan, dtype=np.float32)
    for atlas_zero_based, score in score_per_roi.items():
        out[atlas_data == int(atlas_zero_based) + 1] = float(score)
    return nib.Nifti1Image(out, atlas_img.affine)


# ────────────────────────────────────────────────────────────────────
# Null helpers
# ────────────────────────────────────────────────────────────────────


def _aligned_per_leaf_shift(
    Z_a, leaf_atlas_a, Z_b, leaf_atlas_b
) -> tuple[np.ndarray, list[int]]:
    """Per-leaf cophenetic shift on the intersection of two leaf sets.

    Returns ``(shift_vector, common_atlas_indices)``. The vector aligns
    1-to-1 with the atlas indices."""
    common = sorted(set(leaf_atlas_a.tolist()) & set(leaf_atlas_b.tolist()))
    if len(common) < 4:
        return np.array([], dtype=float), []
    pos_a = {int(r): k for k, r in enumerate(leaf_atlas_a.tolist())}
    pos_b = {int(r): k for k, r in enumerate(leaf_atlas_b.tolist())}
    ia = np.array([pos_a[r] for r in common], dtype=int)
    ib = np.array([pos_b[r] for r in common], dtype=int)
    shift = per_leaf_cophenetic_shift(Z_a, Z_b, ia, ib)
    return shift, common


def strength_preserving_null(
    G_a,
    label_a: str,
    G_b,
    label_b: str,
    *,
    n_surrogates: int,
    cache_dir: Path,
    rng_seed: int = 0,
) -> dict[int, list[float]]:
    """Strength-preserving null distribution of per-ROI cophenetic shifts.

    For ``i`` in 0..n_surrogates-1, generate surrogate-i of ``G_a`` and
    surrogate-i of ``G_b`` (each by independent strength-preserving
    rewiring), build the LRG tree on each, and compute the per-leaf
    shift on the common ROIs. Aggregate across surrogates per atlas
    index.

    Both per-graph surrogate caches and the resulting per-pair null are
    cached on disk, so re-running the same pair is essentially free.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    surrogate_cache = cache_dir / "surrogates"
    pair_cache = cache_dir / "null_pairs"
    pair_cache.mkdir(parents=True, exist_ok=True)
    pair_key = f"{label_a}__vs__{label_b}"
    pair_path = pair_cache / f"{pair_key}__null.pkl"

    if pair_path.exists():
        with open(pair_path, "rb") as f:
            cached = pickle.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("n_surrogates", 0) >= n_surrogates
        ):
            return cached["null_per_atlas"]

    surrogates_a = cached_surrogate_linkages(
        G_a, label_a, n_surrogates,
        surrogate_cache,
        linkage_fn=linkage_at_tau_min,
        rng_seed=rng_seed,
    )
    surrogates_b = cached_surrogate_linkages(
        G_b, label_b, n_surrogates,
        surrogate_cache,
        linkage_fn=linkage_at_tau_min,
        rng_seed=rng_seed + 100_000,
    )

    null_per_atlas: dict[int, list[float]] = {}
    n_eff = min(len(surrogates_a), len(surrogates_b))
    for i in range(n_eff):
        shift, common = _aligned_per_leaf_shift(
            surrogates_a[i]["linkage"],
            surrogates_a[i]["leaf_atlas_idx"],
            surrogates_b[i]["linkage"],
            surrogates_b[i]["leaf_atlas_idx"],
        )
        for k, atlas_idx in enumerate(common):
            null_per_atlas.setdefault(int(atlas_idx), []).append(
                float(shift[k])
            )

    with open(pair_path, "wb") as f:
        pickle.dump(
            {"n_surrogates": n_surrogates, "null_per_atlas": null_per_atlas},
            f,
        )
    return null_per_atlas


def load_subject_ts(
    variant: str, contrast: str, ts_root: Path = TS_BASE
) -> list[np.ndarray]:
    """Load per-subject AFNI .1D TS files for ``(contrast, variant)``."""
    if not ts_root.is_dir():
        return []
    out: list[np.ndarray] = []
    for sub_dir in sorted(ts_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        matches = sorted(
            sub_dir.glob(f"*task-{contrast}_*desc-{variant}.ts.1D")
        )
        if not matches:
            continue
        ts = np.loadtxt(matches[0])
        if ts.ndim != 2 or ts.shape[1] != 100:
            continue
        out.append(ts.astype(np.float64))
    return out


def subject_split_null(
    subject_ts: list[np.ndarray],
    gamma: float,
    *,
    n_splits: int | None = None,
    rng_seed: int = 0,
) -> dict[int, list[float]]:
    """Within-condition balanced split-half null over subjects (legacy)."""
    n_subj = len(subject_ts)
    half = n_subj // 2
    if half < 2:
        return {}
    per_subj_corr = [np.corrcoef(ts.T) for ts in subject_ts]

    all_splits = list(combinations(range(n_subj), half))
    if n_splits is not None and n_splits < len(all_splits):
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(all_splits)
        all_splits = all_splits[:n_splits]

    null_per_atlas: dict[int, list[float]] = {i: [] for i in range(100)}
    for split_a in all_splits:
        set_a = set(split_a)
        split_b = [i for i in range(n_subj) if i not in set_a]
        ma = np.mean(np.stack([per_subj_corr[i] for i in split_a]), axis=0)
        mb = np.mean(np.stack([per_subj_corr[i] for i in split_b]), axis=0)
        try:
            _, _, _, Ga = chain_pipeline(ma, gamma)
            _, _, _, Gb = chain_pipeline(mb, gamma)
            if min(Ga.number_of_nodes(), Gb.number_of_nodes()) < 10:
                continue
            Za, na, _ = linkage_at_tau_min(Ga)
            Zb, nb, _ = linkage_at_tau_min(Gb)
        except RuntimeError:
            continue
        shift, common = _aligned_per_leaf_shift(Za, na, Zb, nb)
        for k, atlas_idx in enumerate(common):
            null_per_atlas.setdefault(int(atlas_idx), []).append(
                float(shift[k])
            )
    return null_per_atlas


# ────────────────────────────────────────────────────────────────────
# Headline figure builder
# ────────────────────────────────────────────────────────────────────


def build_fig3_compare(
    corr_a: np.ndarray,
    label_a: str,
    corr_b: np.ndarray,
    label_b: str,
    *,
    gamma_a: float,
    gamma_b: float | None = None,
    atlas_img,
    atlas_data,
    atlas_labels,
    out_pdf: Path,
    out_csv: Path,
    null_per_atlas: dict[int, list[float]] | None = None,
    null_label: str = "strength-preserving",
) -> dict[str, float]:
    """Render the comparison PDF + per-ROI CSV. See module docstring."""
    if gamma_b is None:
        gamma_b = gamma_a

    _, _, _, G_a = chain_pipeline(corr_a, gamma_a)
    _, _, _, G_b = chain_pipeline(corr_b, gamma_b)
    if min(G_a.number_of_nodes(), G_b.number_of_nodes()) < 4:
        raise RuntimeError("LANS backbone too small for comparison")

    Z_a, nodes_a, _tau_a = linkage_at_tau_min(G_a)
    Z_b, nodes_b, _tau_b = linkage_at_tau_min(G_b)

    common = np.array(
        sorted(set(nodes_a.tolist()) & set(nodes_b.tolist())), dtype=int
    )
    if common.size < 4:
        raise RuntimeError(
            f"only {common.size} common ROIs between {label_a} and {label_b}"
        )

    per_leaf, _ = _aligned_per_leaf_shift(Z_a, nodes_a, Z_b, nodes_b)

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    from scipy.stats import kendalltau, pearsonr

    pos_a = {int(r): i for i, r in enumerate(nodes_a.tolist())}
    pos_b = {int(r): i for i, r in enumerate(nodes_b.tolist())}
    idx_a = np.array([pos_a[r] for r in common.tolist()], dtype=int)
    idx_b = np.array([pos_b[r] for r in common.tolist()], dtype=int)
    Ca = squareform(cophenet(Z_a))[np.ix_(idx_a, idx_a)]
    Cb = squareform(cophenet(Z_b))[np.ix_(idx_b, idx_b)]
    iu = np.triu_indices(Ca.shape[0], k=1)
    bg_common, _ = kendalltau(Ca[iu], Cb[iu])
    cc_common, _ = pearsonr(Ca[iu], Cb[iu])

    null_median = np.full(common.size, np.nan, dtype=float)
    null_p95 = np.full(common.size, np.nan, dtype=float)
    null_n_samples = np.zeros(common.size, dtype=int)
    if null_per_atlas is not None:
        for i, atlas_idx in enumerate(common.tolist()):
            samples = null_per_atlas.get(int(atlas_idx), [])
            if len(samples) >= 3:
                null_median[i] = float(np.median(samples))
                null_p95[i] = float(np.percentile(samples, 95))
                null_n_samples[i] = len(samples)
    calibrated = per_leaf - np.where(
        np.isfinite(null_median), null_median, 0.0
    )

    score_per_roi = {
        int(common[i]): float(per_leaf[i]) for i in range(common.size)
    }
    calibrated_per_roi = {
        int(common[i]): float(calibrated[i])
        for i in range(common.size)
        if np.isfinite(null_median[i])
    }

    rows = []
    for i, atlas_idx in enumerate(common.tolist()):
        rows.append(
            {
                "atlas_index": int(atlas_idx),
                "roi_name": clean_atlas_label(atlas_labels[atlas_idx]),
                "shift_score": float(per_leaf[i]),
                "null_median": (
                    float(null_median[i])
                    if np.isfinite(null_median[i])
                    else np.nan
                ),
                "null_p95": (
                    float(null_p95[i]) if np.isfinite(null_p95[i]) else np.nan
                ),
                "null_n_samples": int(null_n_samples[i]),
                "calibrated_shift": (
                    float(calibrated[i])
                    if np.isfinite(null_median[i])
                    else np.nan
                ),
            }
        )
    sort_col = "calibrated_shift" if null_per_atlas is not None else "shift_score"
    df = pd.DataFrame(rows).sort_values(sort_col, ascending=False)
    df.to_csv(out_csv, index=False)

    # ── Figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 13.0), constrained_layout=False)
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.15],
        width_ratios=[0.75, 1.25],
        hspace=0.30, wspace=0.32,
        left=0.04, right=0.985, top=0.955, bottom=0.05,
    )

    use_calibrated = (
        null_per_atlas is not None and len(calibrated_per_roi) > 0
    )
    if use_calibrated:
        brain_dict = calibrated_per_roi
        brain_values = calibrated[np.isfinite(null_median)]
        vmax_abs = (
            float(np.nanmax(np.abs(brain_values)))
            if brain_values.size
            else 1.0
        )
        vmin_plot, vmax_plot = -vmax_abs, vmax_abs
        brain_cmap = _cmap_transparent_at_center("RdBu_r")
        sentinel_for_bg = 0.0
    else:
        brain_dict = score_per_roi
        brain_values = per_leaf
        vmin_plot = float(np.nanmin(brain_values))
        vmax_plot = float(np.nanmax(brain_values))
        brain_cmap = _cmap_transparent_at_zero("magma_r")
        sentinel_for_bg = 0.0

    brain_img = _shift_to_roi_image(brain_dict, atlas_data, atlas_img)
    arr = np.nan_to_num(brain_img.get_fdata(), nan=sentinel_for_bg)
    brain_img = nib.Nifti1Image(arr.astype(np.float32), atlas_img.affine)

    def _brain_row(ax, mode: str, n_cuts: int, title: str):
        plotting.plot_stat_map(
            brain_img, axes=ax,
            display_mode=mode, cut_coords=n_cuts,
            cmap=brain_cmap, colorbar=True,
            threshold=None,
            vmin=vmin_plot, vmax=vmax_plot,
            symmetric_cbar=use_calibrated,
            annotate=False, draw_cross=False, black_bg=False,
        )
        ax.set_title(title, fontsize=10)

    headline = (
        f"calibrated shift  (+ reorganised, − stable)    "
        f"|max| = {vmax_plot:.3f}    null: {null_label}"
        if use_calibrated
        else f"per-ROI shift  (min = {vmin_plot:.2f}, max = {vmax_plot:.2f})"
    )
    _brain_row(
        fig.add_subplot(gs[0, :]), "z", 7,
        f"{label_a}  vs  {label_b}    axial montage    {headline}",
    )
    _brain_row(fig.add_subplot(gs[1, :]), "x", 7, "sagittal montage")
    _brain_row(fig.add_subplot(gs[2, :]), "y", 7, "coronal montage")

    n_extreme = 10
    bar_col = sort_col
    df_nonnull = df.dropna(subset=[bar_col])
    top_reorg = df_nonnull.head(n_extreme)
    top_stable = df_nonnull.tail(n_extreme)
    selected = pd.concat([top_reorg, top_stable]).reset_index(drop=True)
    bar_colors = (
        ["#B71C1C"] * len(top_reorg) + ["#1565C0"] * len(top_stable)
    )
    selected_plot = selected.iloc[::-1].reset_index(drop=True)
    bar_colors_plot = list(reversed(bar_colors))

    ax_bar = fig.add_subplot(gs[3, 1])
    ax_bar.barh(
        np.arange(len(selected_plot)),
        selected_plot[bar_col].values,
        color=bar_colors_plot, edgecolor="white", linewidth=0.6,
    )
    ax_bar.set_yticks(np.arange(len(selected_plot)))
    ax_bar.set_yticklabels(selected_plot["roi_name"].values, fontsize=8)
    if bar_col == "calibrated_shift":
        ax_bar.set_xlabel(
            "calibrated shift  (obs − null median; > 0 reorganised, < 0 stable)"
        )
        ax_bar.axvline(0.0, color="#90A4AE", lw=0.7, ls="-")
    else:
        ax_bar.set_xlabel(
            "cophenetic-rank shift  (0 = same, 1 = max reorganised)"
        )
    ax_bar.set_title(
        f"Top {len(top_reorg)} reorganised (red)  +  "
        f"Top {len(top_stable)} stable (blue)",
        fontsize=10,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.axhline(len(top_stable) - 0.5, color="#90A4AE", lw=0.8, ls=":")

    ax_hist = fig.add_subplot(gs[3, 0])
    if null_per_atlas is not None:
        pooled_null = (
            np.concatenate(
                [np.asarray(v) for v in null_per_atlas.values() if len(v)]
            )
            if any(len(v) for v in null_per_atlas.values())
            else np.array([])
        )
    else:
        pooled_null = np.array([])
    all_vals = (
        np.concatenate([per_leaf, pooled_null])
        if pooled_null.size
        else per_leaf
    )
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), 26)
    if pooled_null.size:
        ax_hist.hist(
            pooled_null, bins=bin_edges,
            color="#90A4AE", alpha=0.65, edgecolor="white", linewidth=0.4,
            density=True, label=f"null  (n = {pooled_null.size})",
        )
        ax_hist.axvline(
            float(np.median(pooled_null)),
            color="#37474F", lw=1.2, ls="--",
            label=f"null median = {np.median(pooled_null):.3f}",
        )
    ax_hist.hist(
        per_leaf, bins=bin_edges,
        color="#B71C1C", alpha=0.75, edgecolor="white", linewidth=0.4,
        density=True, label=f"observed  (n = {per_leaf.size})",
    )
    ax_hist.axvline(
        float(np.median(per_leaf)),
        color="#B71C1C", lw=1.5, ls="--",
        label=f"obs median = {np.median(per_leaf):.3f}",
    )
    ax_hist.set_xlabel("per-ROI cophenetic shift")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_hist.set_title("Observed vs null shift distribution", fontsize=10)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.suptitle(
        f"fig 3 — dendrogram comparison  ({label_a}  ↔  {label_b})    "
        f"Baker's γ = {bg_common:.3f}    cophenetic r = {cc_common:.3f}    "
        f"|A ∩ B| = {common.size}",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "bakers_gamma_common": float(bg_common),
        "cophenetic_r_common": float(cc_common),
        "n_common": int(common.size),
        "median_shift": float(np.median(per_leaf)),
        "median_null": (
            float(np.nanmedian(null_median))
            if np.isfinite(null_median).any()
            else float("nan")
        ),
        "median_calibrated": (
            float(np.nanmedian(calibrated[np.isfinite(null_median)]))
            if np.isfinite(null_median).any()
            else float("nan")
        ),
    }


def compute_pair(
    corr_a: np.ndarray,
    label_a: str,
    corr_b: np.ndarray,
    label_b: str,
    *,
    gamma_a: float,
    gamma_b: float | None,
    atlas_img,
    atlas_data,
    atlas_labels,
    out_pdf: Path,
    out_csv: Path,
    null_kind: str = "strength-preserving",
    n_surrogates: int = 200,
    cache_dir: Path,
    rng_seed: int = 0,
) -> dict[str, float]:
    """End-to-end pair computation.

    Picks the null source per ``null_kind`` (``"strength-preserving"``,
    ``"subject-split"``, or ``"none"``), then calls
    :func:`build_fig3_compare`.
    """
    if gamma_b is None:
        gamma_b = gamma_a

    null_per_atlas: dict[int, list[float]] | None = None
    null_label = null_kind

    if null_kind == "strength-preserving":
        _, _, _, G_a = chain_pipeline(corr_a, gamma_a)
        _, _, _, G_b = chain_pipeline(corr_b, gamma_b)
        null_per_atlas = strength_preserving_null(
            G_a, label_a, G_b, label_b,
            n_surrogates=n_surrogates,
            cache_dir=cache_dir, rng_seed=rng_seed,
        )
    elif null_kind == "subject-split":
        contrast_a = label_a.split("_", 1)[0]
        contrast_b = label_b.split("_", 1)[0]
        variant_a = label_a[len(contrast_a) + 1 :].rsplit("_", 1)[0]
        variant_b = label_b[len(contrast_b) + 1 :].rsplit("_", 1)[0]
        ts_a = load_subject_ts(variant_a, contrast_a)
        ts_b = load_subject_ts(variant_b, contrast_b)
        merged: dict[int, list[float]] = {i: [] for i in range(100)}
        if len(ts_a) >= 4:
            for k, v in subject_split_null(
                ts_a, gamma_a, rng_seed=rng_seed
            ).items():
                merged[k].extend(v)
        if len(ts_b) >= 4:
            for k, v in subject_split_null(
                ts_b, gamma_b, rng_seed=rng_seed + 1
            ).items():
                merged[k].extend(v)
        if any(merged.values()):
            null_per_atlas = merged
    elif null_kind == "none":
        null_per_atlas = None
    else:
        raise ValueError(f"unknown null_kind: {null_kind!r}")

    return build_fig3_compare(
        corr_a, label_a, corr_b, label_b,
        gamma_a=gamma_a, gamma_b=gamma_b,
        atlas_img=atlas_img, atlas_data=atlas_data,
        atlas_labels=atlas_labels,
        out_pdf=out_pdf, out_csv=out_csv,
        null_per_atlas=null_per_atlas,
        null_label=null_label,
    )
