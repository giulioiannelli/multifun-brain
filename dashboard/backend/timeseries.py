"""Raw-timecourse source for the Signal tab (dataset-aware).

Thin dashboard adapter over :mod:`multifunbrain.io.timeseries` (the canonical
reader). Discovers the ``raw_data_<id>`` directories under ``DATA_ROOT``, exposes
each as a selectable **dataset**, and within the chosen one exposes the
``(subject, contrast, processing)`` facets for the selectors. It memoises both
the file load and the (expensive) EMD sift so repeat views are instant.

Two filename schemes are handled transparently (see the canonical reader): the
BIDS April set (subject/contrast/processing) and the older ``kw…`` sets (subject
from the parent folder, modality token as processing, **no contrast**). For the
kw sets the cohort/per-band analysis is not meaningful (no contrast, no TR), but
the carpet / single-channel / EMD views work.

Query params never build file paths directly — a request resolves to a
*discovered* dataset dir and a *discovered* :class:`TimecourseFile`, so there is
no path-traversal surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from multifunbrain.io import discover_timecourses, load_timecourses, sampling_rate

from . import config
from .atlas import harvard_oxford_names, load_atlas

# Display order mirrors the correlation-side selector (SelectorBar PROC_ORDER).
_PROC_ORDER = [
    "bpfBOLD", "bpfVASO", "MIRNoise_bold", "optcom_bold", "optcomMIRDenoised_bold",
]


def _order_by(values, order):
    def key(v):
        return (order.index(v) if v in order else len(order), v)

    return sorted(values, key=key)


# --- dataset discovery (raw_data_<id> dirs under DATA_ROOT) ------------------


def _raw_data_dirs() -> list[Path]:
    """All ``raw_data_*`` directories under ``DATA_ROOT`` (plus the default)."""
    dirs: list[Path] = []
    root = config.DATA_ROOT
    if root.is_dir():
        dirs = sorted(p for p in root.glob("raw_data_*") if p.is_dir())
    default = config.RAW_DATA_ROOT
    if default.is_dir() and default not in dirs:
        dirs.insert(0, default)
    return dirs


def _dataset_id(path: Path) -> str:
    name = path.name
    return name[len("raw_data_") :] if name.startswith("raw_data_") else name


def _root_for(dataset: str | None) -> Path:
    """Resolve a dataset id to its directory (validated), else the default root."""
    if dataset:
        for d in _raw_data_dirs():
            if _dataset_id(d) == dataset:
                return d
    return config.RAW_DATA_ROOT


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_dir() else 0.0


def raw_datasets() -> list[dict]:
    """List available raw datasets (powers the Signal-tab Dataset dropdown)."""
    default_id = _dataset_id(config.RAW_DATA_ROOT)
    out: list[dict] = []
    for d in _raw_data_dirs():
        files = discover_timecourses(d)
        if not files:
            continue
        subjects = {f.subject for f in files}
        contrasts = {f.contrast for f in files if f.contrast}
        out.append(
            {
                "id": _dataset_id(d),
                "label": _dataset_id(d),
                "n_subjects": len(subjects),
                "n_files": len(files),
                "has_contrast": bool(contrasts),
            }
        )
    out.sort(key=lambda e: (e["id"] != default_id, e["id"]))  # default first
    return out


# --- per-dataset file index + loads -----------------------------------------


@lru_cache(maxsize=32)
def _load_cached(path: str, _mtime: float) -> np.ndarray:
    return load_timecourses(path)  # (n_regions, n_timepoints)


def _safe_load(path: str) -> np.ndarray | None:
    """Load a timecourse, or ``None`` if it's empty/malformed (never raises)."""
    try:
        return _load_cached(path, Path(path).stat().st_mtime)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=8)
def _index(root_str: str, _mtime: float) -> dict:
    """Map ``(subject, contrast, processing) -> path`` for one dataset root.

    ``contrast`` is ``""`` for the kw scheme (no co2/rest).
    """
    files = discover_timecourses(Path(root_str))
    return {(f.subject, f.contrast or "", f.processing): str(f.path) for f in files}


def _resolve(root: Path, subject: str, contrast: str, processing: str) -> str | None:
    return _index(str(root), _mtime(root)).get((subject, contrast or "", processing))


@lru_cache(maxsize=8)
def _n_regions(root_str: str, _mtime: float) -> int | None:
    """Region (channel) count for a dataset, from the first file that loads."""
    for f in discover_timecourses(Path(root_str)):
        ts = _safe_load(str(f.path))
        if ts is not None:
            return int(ts.shape[0])
    return None


@lru_cache(maxsize=8)
def _atlas_for(root_str: str, _mtime: float) -> str | None:
    """Atlas id for a dataset, from the discovered filenames' prefix
    (``schaefer100`` / ``harvardoxford48``); ``None`` if unrecognised."""
    for f in discover_timecourses(Path(root_str)):
        if f.atlas:
            return f.atlas
    return None


def region_names(dataset: str | None = None) -> list[str]:
    """Channel labels for the chosen dataset's atlas.

    Resolves the atlas from the discovered filenames (``schaefer100`` /
    ``harvardoxford48``), then by region count, then falls back to generic
    ``region-N``. The fallback covers the case where the HarvardOxford label XML
    is absent (``data/`` is gitignored, so it ships with the data, not the repo).
    """
    root = _root_for(dataset)
    n = _n_regions(str(root), _mtime(root))
    atlas = _atlas_for(str(root), _mtime(root))
    schaefer = [r.short for r in load_atlas()]

    # HarvardOxford-48 (kw set): names from the FSL XML, when present.
    if atlas == "harvardoxford48" or n == 48:
        ho = list(harvard_oxford_names())
        if ho and (n is None or len(ho) == n):
            return ho

    # Schaefer-100 (April + Nov-2025): the order-file names.
    if atlas == "schaefer100" or n is None or n == len(schaefer):
        return schaefer

    return [f"region-{i}" for i in range(n)]


def signal_catalog(dataset: str | None = None) -> dict:
    """Available raw timecourses + facets for the chosen dataset (+ dataset list)."""
    root = _root_for(dataset)
    index = _index(str(root), _mtime(root))
    entries = [
        {"subject": s, "contrast": c, "processing": p} for (s, c, p) in index
    ]
    entries.sort(key=lambda e: (e["subject"], e["contrast"], e["processing"]))
    subjects = sorted({e["subject"] for e in entries})
    contrasts = sorted({e["contrast"] for e in entries if e["contrast"]})
    processings = _order_by({e["processing"] for e in entries}, _PROC_ORDER)
    names = region_names(dataset)
    return {
        "dataset": _dataset_id(root),
        "datasets": raw_datasets(),
        "has_contrast": bool(contrasts),
        "entries": entries,
        "subjects": subjects,
        "contrasts": contrasts,
        "processings": processings,
        "region_names": names,
        "n_regions": len(names),
        "root": str(root),
    }


def get_timecourses(
    dataset: str | None, subject: str, contrast: str, processing: str
) -> np.ndarray | None:
    """Region-major ``(n_regions, n_timepoints)`` array, or ``None`` if unknown."""
    path = _resolve(_root_for(dataset), subject, contrast, processing)
    if path is None:
        return None
    return _safe_load(path)


@lru_cache(maxsize=128)
def _sift_cached(path: str, _mtime: float, channel: int, sample_rate: float) -> dict:
    """Full EMD sift of one channel (all IMFs) + per-IMF characteristic freqs.

    Cached because sift is the expensive step; shared by the EMD panel and the
    per-band reconstruction. ``imfs`` is ``(n_timepoints, n_imf)``; ``freqs`` is
    the median instantaneous frequency per IMF (Hz when ``sample_rate = 1/TR``;
    cycles/sample when ``sample_rate=1`` — unknown variants, e.g. the kw sets).
    """
    from multifunbrain.analysis.emd_bands import sift_with_frequencies

    ts = _load_cached(path, _mtime)
    n_regions, _ = ts.shape
    channel = int(np.clip(channel, 0, n_regions - 1))
    x = np.asarray(ts[channel], dtype=float)
    imfs, freqs = sift_with_frequencies(x, sample_rate)  # (T, n_imf), (n_imf,)
    return {
        "channel": channel,
        "n_imf": int(imfs.shape[1]),
        "imfs": imfs,  # (n_timepoints, n_imf)
        "freqs": freqs,
        "signal": x,
    }


def sift_channel(
    dataset: str | None,
    subject: str,
    contrast: str,
    processing: str,
    channel: int,
    max_imfs: int = 10,
) -> dict | None:
    """EMD IMF decomposition of one channel for display, or ``None`` if unknown.

    Caps to ``max_imfs`` IMFs (the residual carries the rest) and returns IMFs as
    rows for the stacked figure.
    """
    path = _resolve(_root_for(dataset), subject, contrast, processing)
    if path is None:
        return None
    fs = sampling_rate(processing)
    try:
        full = _sift_cached(path, Path(path).stat().st_mtime, int(channel), fs)
    except (OSError, ValueError):
        return None
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
def _cohort_pool_cached(
    root_str: str, contrast: str, processing: str, _mtime: float
) -> dict:
    """Pool every IMF's characteristic frequency (Hz) across the cohort.

    Runs EMD over every ROI of every subject for ``(contrast, processing)`` —
    expensive (~3 s for 6 subjects × 100 ROIs), hence cached. Scheme-independent.
    """
    from multifunbrain.analysis.emd_bands import sift_with_frequencies

    files = discover_timecourses(
        Path(root_str), contrasts=[contrast], processings=[processing]
    )
    fs = sampling_rate(processing)
    per_index: dict[int, list[float]] = {}
    all_freqs: list[float] = []
    for fobj in files:
        ts = _safe_load(str(fobj.path))
        if ts is None:
            continue
        for roi in ts:
            _, freqs = sift_with_frequencies(roi, fs)  # Hz, median IF
            for k, fr in enumerate(freqs):
                if np.isfinite(fr) and fr > 0:
                    per_index.setdefault(k, []).append(float(fr))
                    all_freqs.append(float(fr))
    return {
        "contrast": contrast,
        "processing": processing,
        "sample_rate": fs,
        "n_subjects": len(files),
        "all_freqs": np.asarray(all_freqs, dtype=float),
        "per_index": {k: np.asarray(v, dtype=float) for k, v in per_index.items()},
    }


def _index_cluster_medians(per_index: dict, min_count: int = 5) -> dict:
    """Per-IMF-index median frequency (Hz) — for the cluster annotations."""
    out: dict[int, float] = {}
    for k, vals in per_index.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size >= min_count:
            out[int(k)] = float(np.median(arr))
    return out


def _scheme_for(pool: dict, scheme: str):
    """Build the band scheme for *pool* under ``"canonical"`` or ``"data_driven"``."""
    from multifunbrain.analysis.emd_bands import canonical_scheme, estimate_band_edges

    if scheme == "data_driven":
        if not pool["per_index"]:
            return None
        try:
            return estimate_band_edges(pool["per_index"])
        except ValueError:
            return None
    # canonical (default): fixed Hz edges, cluster medians kept for annotation.
    return canonical_scheme(_index_cluster_medians(pool["per_index"]))


def cohort_bands(
    dataset: str | None, contrast: str, processing: str, scheme: str = "canonical"
) -> dict | None:
    """Cohort IMF-frequency pooling (Hz) + the chosen band scheme, or ``None``.

    ``scheme`` is ``"canonical"`` (fixed slow-oscillation edges, default) or
    ``"data_driven"``. The pooled histogram is identical for both — only the band
    overlay/assignment differs.
    """
    root = _root_for(dataset)
    if not discover_timecourses(root, contrasts=[contrast], processings=[processing]):
        return None
    pool = _cohort_pool_cached(str(root), contrast, processing, _mtime(root))
    return {**pool, "scheme": _scheme_for(pool, scheme), "scheme_name": scheme}


def band_reconstruction(
    dataset: str | None,
    subject: str,
    contrast: str,
    processing: str,
    channel: int,
    scheme: str = "canonical",
) -> dict | None:
    """Per-band reconstructed timecourses of one channel (the per-band signals).

    Sums each band's IMFs for the selected channel using the chosen scheme's
    edges. Canonical edges are fixed, so that path skips the (expensive) cohort
    pool; ``data_driven`` derives the edges from the cohort first.
    """
    root = _root_for(dataset)
    path = _resolve(root, subject, contrast, processing)
    if path is None:
        return None
    from multifunbrain.analysis.emd_bands import canonical_scheme, reconstruct_bands

    if scheme == "data_driven":
        cohort = cohort_bands(dataset, contrast, processing, "data_driven")
        if cohort is None or cohort["scheme"] is None:
            return None
        band_scheme = cohort["scheme"]
    else:
        band_scheme = canonical_scheme()

    fs = sampling_rate(processing)
    try:
        full = _sift_cached(path, Path(path).stat().st_mtime, int(channel), fs)
    except (OSError, ValueError):
        return None
    bands = band_scheme.bands
    signals, assignment = reconstruct_bands(full["imfs"], full["freqs"], bands)
    return {
        "channel": full["channel"],
        "signal": full["signal"],
        "freqs": full["freqs"],
        "bands": bands,
        "signals": signals,
        "assignment": assignment,
        "scheme": scheme,
        "sample_rate": fs,
    }
