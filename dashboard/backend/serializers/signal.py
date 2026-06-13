"""Serializers for the Signal tab (raw ROI timecourses).

Unlike the Section-1/3 serializers (keyed by ``PipelineResult``), these operate
on raw ``(n_regions, n_timepoints)`` arrays loaded by
:mod:`dashboard.backend.timeseries`. The route loads the array and passes it in;
serializers stay pure (array + atlas names -> JSON-safe dict).
"""

from __future__ import annotations

import numpy as np

from ..encode import clean


def _names_for(n: int, names: list[str]) -> list[str]:
    """Atlas short names if they line up with the row count, else generic labels."""
    if names and len(names) == n:
        return list(names)
    return [f"region-{i}" for i in range(n)]


def heatmap_spec(ts: np.ndarray, names: list[str], **_) -> dict:
    """Carpet/grayplot payload: the full ``(n_regions, n_timepoints)`` matrix.

    Per-region normalisation (z-score) is applied client-side via a toggle, so
    the raw values are shipped as-is (true values also drive the hover).
    """
    ts = np.asarray(ts, dtype=float)
    n_regions, n_timepoints = ts.shape
    return clean(
        {
            "kind": "signal_heatmap",
            "z": ts,
            "names": _names_for(n_regions, names),
            "n_regions": int(n_regions),
            "n_timepoints": int(n_timepoints),
        }
    )


def channel_spec(ts: np.ndarray, names: list[str], channel: int = 0, **_) -> dict:
    """Single-channel timecourse: value per timepoint for one region."""
    ts = np.asarray(ts, dtype=float)
    n_regions, n_timepoints = ts.shape
    ch = int(np.clip(channel, 0, n_regions - 1))
    labels = _names_for(n_regions, names)
    return clean(
        {
            "kind": "signal_channel",
            "channel": ch,
            "name": labels[ch],
            "n_regions": int(n_regions),
            "t": list(range(n_timepoints)),
            "y": ts[ch],
        }
    )


def cohort_bands_spec(cohort: dict, **_) -> dict:
    """Cohort IMF-frequency histogram + the data-driven s5/s4/s* band ranges.

    Reconstructs Daniele's frequency histogram (in cycles/sample): pooled per-IMF
    characteristic frequencies across all subjects/ROIs, with the band ranges and
    per-IMF-index cluster centres for shading/annotation.
    """
    freqs = np.asarray(cohort["all_freqs"], dtype=float)
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    scheme = cohort["scheme"]
    if freqs.size < 2 or scheme is None:
        return {"kind": "cohort_bands", "error": "not enough IMF frequencies to build the histogram"}

    bands = {name: list(rng) for name, rng in scheme.bands.items()}
    hi = max(float(np.percentile(freqs, 99.7)), bands["sstar"][1])
    counts, edges = np.histogram(freqs, bins=64, range=(0.0, hi))
    centers = {int(k): float(v) for k, v in scheme.centers.items()}
    # IMFs per index that contributed (cluster sizes), for hover/context.
    cluster_n = {int(k): int(np.size(v)) for k, v in cohort["per_index"].items()}
    return clean(
        {
            "kind": "cohort_bands",
            "contrast": cohort["contrast"],
            "processing": cohort["processing"],
            "n_subjects": int(cohort["n_subjects"]),
            "n_imf_total": int(freqs.size),
            "counts": counts,
            "edges": edges,
            "bands": bands,
            "drift_max": float(scheme.drift_max),
            "cluster_centers": centers,
            "cluster_n": cluster_n,
        }
    )


def band_reconstruction_spec(recon: dict, names: list[str], **_) -> dict:
    """Per-band reconstructed timecourses for one channel (the per-band signals)."""
    signal = np.asarray(recon["signal"], dtype=float)
    n_t = signal.size
    ch = int(recon["channel"])
    labels = _names_for(len(names) or ch + 1, names)
    name = labels[ch] if ch < len(labels) else f"region-{ch}"
    freqs = np.asarray(recon["freqs"], dtype=float)
    bands = {b: list(rng) for b, rng in recon["bands"].items()}
    signals = {b: np.asarray(s, dtype=float) for b, s in recon["signals"].items()}
    # IMF indices feeding each band + their characteristic frequencies.
    assignment = {
        b: [{"imf": int(i), "freq": float(freqs[i]) if i < freqs.size else None} for i in idx]
        for b, idx in recon["assignment"].items()
    }
    return clean(
        {
            "kind": "signal_bands",
            "channel": ch,
            "name": name,
            "t": list(range(n_t)),
            "signal": signal,
            "bands": bands,
            "signals": signals,
            "assignment": assignment,
        }
    )


def emd_spec(sift: dict, names: list[str], **_) -> dict:
    """EMD IMF decomposition of one channel: original signal + IMFs + residual.

    ``sift`` is the memoised output of :func:`timeseries.sift_channel`
    (``imfs`` is ``(n_imf, n_timepoints)``).
    """
    signal = np.asarray(sift["signal"], dtype=float)
    imfs = np.asarray(sift["imfs"], dtype=float)
    residual = np.asarray(sift["residual"], dtype=float)
    ch = int(sift["channel"])
    n_t = signal.size
    labels = _names_for(len(names) or ch + 1, names)
    name = labels[ch] if ch < len(labels) else f"region-{ch}"
    return clean(
        {
            "kind": "signal_emd",
            "channel": ch,
            "name": name,
            "n_imf": int(sift["n_imf"]),
            "t": list(range(n_t)),
            "signal": signal,
            "imfs": imfs,  # one row per IMF
            "residual": residual,
        }
    )
