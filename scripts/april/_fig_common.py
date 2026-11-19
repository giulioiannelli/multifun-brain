"""Shared figure-building helpers for scripts/april (07/08/09).

Single canonical home for:

- :func:`gamma_for_variant` — Marchenko-Pastur aspect ratio per processing variant.
- :func:`chain_pipeline` — raw → prepare → MP-denoise → clip-positive → LANS.
- :func:`linkage_at_tau_min` — LRG diffusion linkage at τ_min = 1/λ_max.
- :func:`load_atlas` — Schaefer 2018 100-parcel × 17-network atlas (cached).
- Constants for the April batch layout (paths, contrasts, variants, bands).

07 and 08 used to keep their own copies of these helpers; lifting them
here keeps the pipeline definition in one place and lets 09's batch
driver call the same pipeline as the figure scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import nibabel as nib
import numpy as np

# ``linkage_at_tau_min`` now lives in the library (single canonical home); kept
# importable here so 08/09/_handoff_compare keep their ``from _fig_common import``.
from multifunbrain.analysis.lrg.partitions import linkage_at_tau_min  # noqa: F401

ROOT = Path("/home/opisthofulax/Documents/research+/brain_network/multifun-brain")
SRC_BASE = ROOT / "data/correlation_mat_april_data"
TS_BASE = SRC_BASE / "20260326_raw-data_Maria"
RESULTS_BASE = ROOT / "data/correlation_mat_april_data_results"

CONTRASTS = ("co2", "rest")
PROCESSING_VARIANTS = (
    "bpfBOLD",
    "bpfVASO",
    "MIRNoise_bold",
    "optcom_bold",
    "optcomMIRDenoised_bold",
)
BANDS = ("s4", "s5", "sstar")


def gamma_for_variant(variant: str) -> float:
    """Marchenko-Pastur aspect ratio ``gamma = n_parcels / n_timepoints``.

    Matches the per-processing TR counts in
    :data:`multifunbrain.datasets.april.N_TIMEPOINTS_PER_PROCESSING`:
    bpfBOLD / bpfVASO → 100 / 442, optcom* / MIRNoise → 100 / 597.
    """
    if "bpfBOLD" in variant or "bpfVASO" in variant:
        return 100 / 442
    return 100 / 597


def chain_pipeline(
    corr_raw: np.ndarray,
    gamma: float,
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
    """raw → prepare → MP-denoise → clip-positive → LANS backbone.

    Returns
    -------
    corr_prep : np.ndarray
        After NaN-row / dead-region drop.
    corr_mp : np.ndarray
        After Marchenko-Pastur shrinkage.
    corr_pos : np.ndarray
        ``np.clip(corr_mp, 0, None)`` with the diagonal zeroed.
    G_lans : nx.Graph
        LANS-validated backbone (positive weights only).
    """
    from multifunbrain.preprocessing.denoising import marchenko_pastur_denoise
    from multifunbrain.preprocessing.prepare import prepare_correlation_matrix
    from multifunbrain.processing.backbone import filter_validated

    corr_prep = prepare_correlation_matrix(np.asarray(corr_raw, dtype=float))
    corr_mp = marchenko_pastur_denoise(corr_prep, gamma=gamma, sigma=1.0)
    corr_pos = np.clip(corr_mp, 0.0, None)
    np.fill_diagonal(corr_pos, 0.0)
    G_lans, _ = filter_validated(
        corr_pos, method="lans", alpha=alpha, weights="positive"
    )
    return corr_prep, corr_mp, corr_pos, G_lans


_ATLAS_CACHE: dict[str, Any] = {}


def load_atlas(n_rois: int = 100, yeo_networks: int = 17) -> tuple[Any, np.ndarray, list[str]]:
    """Schaefer 2018 atlas (cached).

    Returns ``(nib.Nifti1Image, atlas_data int array, labels list[str])``.
    Subsequent calls with the same parameters re-use the cached values.
    """
    from nilearn import datasets

    key = f"{n_rois}_{yeo_networks}"
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=yeo_networks
    )
    img = nib.load(atlas["maps"])
    data = img.get_fdata().astype(int)
    labels = [
        b.decode("utf-8") if isinstance(b, bytes) else str(b)
        for b in atlas["labels"]
    ]
    _ATLAS_CACHE[key] = (img, data, labels)
    return img, data, labels


def clean_atlas_label(raw: str) -> str:
    """Strip Schaefer's ``17Networks_`` prefix from a label."""
    if raw.startswith("17Networks_"):
        return raw[len("17Networks_"):]
    return raw
