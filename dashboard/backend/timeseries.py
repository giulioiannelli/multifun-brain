"""Raw-timecourse source for the Signal tab.

Thin dashboard adapter over :mod:`multifunbrain.io.timeseries` (the canonical
reader). Discovers ``.ts.1D`` files under ``RAW_DATA_ROOT``, exposes their
``(subject, contrast, processing)`` facets for the selectors, and memoises both
the file load and the (expensive) EMD sift so repeat views are instant.

Query params never build file paths directly — a request resolves to a
*discovered* :class:`TimecourseFile`, so there is no path-traversal surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from multifunbrain.io import discover_timecourses, load_timecourses

from . import config
from .atlas import load_atlas

# Display order mirrors the correlation-side selector (SelectorBar PROC_ORDER).
_PROC_ORDER = [
    "bpfBOLD", "bpfVASO", "MIRNoise_bold", "optcom_bold", "optcomMIRDenoised_bold",
]


def _order_by(values, order):
    def key(v):
        return (order.index(v) if v in order else len(order), v)

    return sorted(values, key=key)


@lru_cache(maxsize=1)
def _index(_mtime: float) -> dict:
    """Map ``(subject, contrast, processing) -> path`` for all discovered files."""
    files = discover_timecourses(config.RAW_DATA_ROOT)
    return {(f.subject, f.contrast, f.processing): str(f.path) for f in files}


def _root_mtime() -> float:
    root = config.RAW_DATA_ROOT
    return root.stat().st_mtime if root.is_dir() else 0.0


def _resolve(subject: str, contrast: str, processing: str) -> str | None:
    return _index(_root_mtime()).get((subject, contrast, processing))


def region_names() -> list[str]:
    """Atlas short names for the parcels (channel labels)."""
    return [r.short for r in load_atlas()]


def signal_catalog() -> dict:
    """Available raw timecourses + facets, for the Signal selectors."""
    index = _index(_root_mtime())
    entries = [
        {"subject": s, "contrast": c, "processing": p}
        for (s, c, p) in index
    ]
    entries.sort(key=lambda e: (e["subject"], e["contrast"], e["processing"]))
    subjects = sorted({e["subject"] for e in entries})
    contrasts = sorted({e["contrast"] for e in entries})
    processings = _order_by({e["processing"] for e in entries}, _PROC_ORDER)
    return {
        "entries": entries,
        "subjects": subjects,
        "contrasts": contrasts,
        "processings": processings,
        "region_names": region_names(),
        "n_regions": len(region_names()),
        "root": str(config.RAW_DATA_ROOT),
    }


@lru_cache(maxsize=32)
def _load_cached(path: str, _mtime: float) -> np.ndarray:
    return load_timecourses(path)  # (n_regions, n_timepoints)


def get_timecourses(subject: str, contrast: str, processing: str) -> np.ndarray | None:
    """Region-major ``(n_regions, n_timepoints)`` array, or ``None`` if unknown."""
    path = _resolve(subject, contrast, processing)
    if path is None:
        return None
    return _load_cached(path, Path(path).stat().st_mtime)


@lru_cache(maxsize=128)
def _sift_cached(path: str, _mtime: float, channel: int) -> dict:
    """Full EMD sift of one channel (all IMFs) + per-IMF characteristic freqs.

    Cached because sift is the expensive step; shared by the EMD panel and the
    per-band reconstruction. ``imfs`` is ``(n_timepoints, n_imf)``;
    ``freqs`` is cycles/sample.
    """
    from multifunbrain.analysis.emd_bands import sift_with_frequencies

    ts = _load_cached(path, _mtime)
    n_regions, _ = ts.shape
    channel = int(np.clip(channel, 0, n_regions - 1))
    x = np.asarray(ts[channel], dtype=float)
    imfs, freqs = sift_with_frequencies(x, 1.0)  # (T, n_imf), (n_imf,)
    return {
        "channel": channel,
        "n_imf": int(imfs.shape[1]),
        "imfs": imfs,  # (n_timepoints, n_imf)
        "freqs": freqs,
        "signal": x,
    }


def sift_channel(
    subject: str, contrast: str, processing: str, channel: int, max_imfs: int = 10
) -> dict | None:
    """EMD IMF decomposition of one channel for display, or ``None`` if unknown.

    Caps to ``max_imfs`` IMFs (the residual carries the rest) and returns IMFs as
    rows for the stacked figure.
    """
    path = _resolve(subject, contrast, processing)
    if path is None:
        return None
    full = _sift_cached(path, Path(path).stat().st_mtime, int(channel))
    imfs = full["imfs"]  # (T, n_imf)
    x = full["signal"]
    if max_imfs and imfs.shape[1] > max_imfs:
        imfs = imfs[:, :max_imfs]
    residual = x - imfs.sum(axis=1)
    return {
        "channel": full["channel"],
        "n_imf": int(imfs.shape[1]),
        "imfs": imfs.T,  # (n_imf, n_timepoints) — one row per IMF
        "residual": residual,
        "signal": x,
    }


@lru_cache(maxsize=16)
def _cohort_bands_cached(contrast: str, processing: str, _mtime: float) -> dict:
    """Pool per-IMF characteristic freqs across the cohort + estimate band edges.

    Runs EMD over every ROI of every subject for ``(contrast, processing)`` —
    expensive (~3 s for 6 subjects × 100 ROIs), hence cached. Returns the pooled
    frequencies, per-IMF-index clusters, and the data-driven :class:`BandScheme`.
    """
    from multifunbrain.analysis.emd_bands import (
        estimate_band_edges,
        sift_with_frequencies,
    )

    files = discover_timecourses(
        config.RAW_DATA_ROOT, contrasts=[contrast], processings=[processing]
    )
    per_index: dict[int, list[float]] = {}
    all_freqs: list[float] = []
    for fobj in files:
        ts = _load_cached(str(fobj.path), fobj.path.stat().st_mtime)
        for roi in ts:
            _, freqs = sift_with_frequencies(roi, 1.0)
            for k, fr in enumerate(freqs):
                if np.isfinite(fr) and fr > 0:
                    per_index.setdefault(k, []).append(float(fr))
                    all_freqs.append(float(fr))

    scheme = None
    if per_index:
        try:
            scheme = estimate_band_edges(per_index)
        except ValueError:
            scheme = None
    return {
        "contrast": contrast,
        "processing": processing,
        "n_subjects": len(files),
        "all_freqs": np.asarray(all_freqs, dtype=float),
        "per_index": {k: np.asarray(v, dtype=float) for k, v in per_index.items()},
        "scheme": scheme,
    }


def cohort_bands(contrast: str, processing: str) -> dict | None:
    """Cohort IMF-frequency pooling + band scheme, or ``None`` if no files."""
    if not discover_timecourses(
        config.RAW_DATA_ROOT, contrasts=[contrast], processings=[processing]
    ):
        return None
    return _cohort_bands_cached(contrast, processing, _root_mtime())


def band_reconstruction(
    subject: str, contrast: str, processing: str, channel: int
) -> dict | None:
    """Per-band reconstructed timecourses of one channel (the per-band signals).

    Uses the cohort band scheme for ``(contrast, processing)`` so the bands match
    the cohort histogram, then sums each band's IMFs for the selected channel.
    """
    path = _resolve(subject, contrast, processing)
    if path is None:
        return None
    cohort = cohort_bands(contrast, processing)
    if cohort is None or cohort["scheme"] is None:
        return None
    from multifunbrain.analysis.emd_bands import reconstruct_bands

    full = _sift_cached(path, Path(path).stat().st_mtime, int(channel))
    bands = cohort["scheme"].bands
    signals, assignment = reconstruct_bands(full["imfs"], full["freqs"], bands)
    return {
        "channel": full["channel"],
        "signal": full["signal"],
        "freqs": full["freqs"],
        "bands": bands,
        "signals": signals,
        "assignment": assignment,
    }
