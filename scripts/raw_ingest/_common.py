"""Turn a raw ``raw_data_<id>`` timecourse tree into dashboard result bundles.

For each ``(subject, variant)`` timecourse this decomposes every ROI into the
canonical frequency bands (EMD), builds the broadband + per-band correlation
matrices (:func:`multifunbrain.analysis.emd_bands.band_correlation_matrices`),
and runs the standard pipeline on each — producing the exact same
``global`` / ``bands`` / ``patients`` bundle layout the April batch writes, so the
dashboard treats a computed kw dataset identically to April.

- **Per subject**: one result per ``(subject, variant, band)``.
- **Group mean**: per ``(variant, band)``, the mean of that variant's per-subject
  correlation matrices (subjects of a variant share N, so the MP ``gamma`` is
  well defined) → one broadband ``global`` result and one per-band result.
- Labels mirror April but carry **no contrast** (the kw sets have none):
  ``global/<variant>``, ``band/<variant>/<band>``,
  ``patient/<subj>/<variant>[/<band>]``.

Degenerate band matrices (a band above a slow acquisition's Nyquist limit yields
all-zero band signals → a constant/NaN correlation) are recorded as graceful
per-matrix failures (``result.error``), never crashes.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from multifunbrain import PipelineConfig, PipelineResult, run_pipeline
from multifunbrain.analysis.emd_bands import BAND_ORDER, band_correlation_matrices
from multifunbrain.io import discover_timecourses, load_timecourses, sampling_rate_for

log = logging.getLogger(__name__)

# Broadband + the canonical bands (low → high): full, s5, s4, sstar.
BANDS: tuple[str, ...] = ("full", *BAND_ORDER)


def _label(level: str, subject: str | None, variant: str, band: str) -> str:
    """Pipeline label mirroring April's scheme (contrast-less for kw datasets)."""
    if level == "patient":
        base = f"patient/{subject}/{variant}"
        return base if band == "full" else f"{base}/{band}"
    if band == "full":
        return f"global/{variant}"
    return f"band/{variant}/{band}"


def _run(corr: np.ndarray, gamma: float | None, label: str) -> PipelineResult:
    """Run the pipeline on one matrix, downgrading any failure to a recorded error."""
    cfg = PipelineConfig(gamma=gamma)  # repo defaults: percolation filter, LRG on, no Louvain
    try:
        return run_pipeline(np.asarray(corr, dtype=float), config=cfg, label=label)
    except Exception as exc:  # noqa: BLE001 - one bad matrix must not abort the batch
        log.warning("[%s] pipeline failed: %s", label, exc)
        return PipelineResult(config=cfg, label=label, error=f"{type(exc).__name__}: {exc}")


def plan_size(raw_root: str | Path) -> int:
    """Total pipeline runs the ingest will do (per-subject + group means)."""
    files = discover_timecourses(Path(raw_root))
    variants = {f.processing for f in files}
    return len(files) * len(BANDS) + len(variants) * len(BANDS)


def ingest_dataset(
    raw_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[PipelineResult]]:
    """Compute all bundles for one raw dataset.

    Returns ``{"global": [...], "bands": [...], "patients": [...]}`` — the three
    result lists ready for :func:`dump_bundle`. ``progress`` (if given) is called
    with each result's label as it completes.
    """
    raw_root = Path(raw_root)
    files = discover_timecourses(raw_root)
    if not files:
        raise ValueError(f"no timecourses found under {raw_root}")

    by_variant: dict[str, dict[str, object]] = defaultdict(dict)
    for f in files:
        by_variant[f.processing][f.subject] = f  # one file per (variant, subject)

    out: dict[str, list[PipelineResult]] = {"global": [], "bands": [], "patients": []}

    for variant in sorted(by_variant):
        subj_files = by_variant[variant]
        accum: dict[str, list[np.ndarray]] = {b: [] for b in BANDS}
        gamma_variant: float | None = None

        for subject in sorted(subj_files):
            fobj = subj_files[subject]
            try:
                ts = load_timecourses(fobj.path)  # (n_regions, n_timepoints)
            except (OSError, ValueError) as exc:
                log.warning("[%s/%s] load failed: %s", subject, variant, exc)
                continue
            n_regions, n_t = ts.shape
            gamma = n_regions / n_t if n_t else None
            gamma_variant = gamma  # subjects of a variant share N → shared gamma
            fs = sampling_rate_for(raw_root, n_t, variant)
            mats = band_correlation_matrices(ts, fs)  # {full, s5, s4, sstar}
            for band in BANDS:
                corr = np.asarray(mats[band], dtype=float)
                accum[band].append(corr)
                label = _label("patient", subject, variant, band)
                out["patients"].append(_run(corr, gamma, label))
                if progress:
                    progress(label)

        # Per-variant group means (broadband → global bundle, bands → bands bundle).
        for band in BANDS:
            if not accum[band]:
                continue
            mean_corr = np.nanmean(np.stack(accum[band]), axis=0)
            label = _label("global", None, variant, band)
            result = _run(mean_corr, gamma_variant, label)
            out["global" if band == "full" else "bands"].append(result)
            if progress:
                progress(label)

    return out


# ──────────────────────────────────────────────────────────────────────
# Output (self-contained: same on-disk shape as scripts/april/_common.py)
# ──────────────────────────────────────────────────────────────────────


def dump_bundle(out_dir: Path, results: Iterable[PipelineResult], config: PipelineConfig) -> None:
    """Write ``results.pkl`` + ``summary.csv`` + ``config.json`` + ``failed_matrices.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)

    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    frames = [r.summary_table() for r in results if r.network_analyses]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "summary.csv", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    failures = [{"label": r.label, "error": r.error} for r in results if r.error is not None]
    with open(out_dir / "failed_matrices.json", "w") as f:
        json.dump(failures, f, indent=2)
