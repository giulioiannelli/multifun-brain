"""Serializers for the Signal tab (raw ROI timecourses).

Unlike the Section-1/3 serializers (keyed by ``PipelineResult``), these operate
on raw ``(n_regions, n_timepoints)`` arrays loaded by
:mod:`dashboard.backend.timeseries`. The route loads the array and passes it in;
serializers stay pure (array + atlas names -> JSON-safe dict).
"""

from __future__ import annotations

import numpy as np

from ..encode import clean

# Label for the mean-over-regions pseudo-channel (channel index -1).
AVG_NAME = "⟨mean over regions⟩"


def _names_for(n: int, names: list[str]) -> list[str]:
    """Atlas short names if they line up with the row count, else generic labels."""
    if names and len(names) == n:
        return list(names)
    return [f"region-{i}" for i in range(n)]


def _channel_name(ch: int, labels: list[str]) -> str:
    """Resolve a channel index to its label; ``-1`` is the mean-over-regions signal."""
    if ch < 0:
        return AVG_NAME
    return labels[ch] if ch < len(labels) else f"region-{ch}"


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
    """Single-channel timecourse: value per timepoint for one region.

    ``channel = -1`` returns the mean timecourse over all regions (the global
    average signal).
    """
    ts = np.asarray(ts, dtype=float)
    n_regions, n_timepoints = ts.shape
    labels = _names_for(n_regions, names)
    if channel < 0:
        ch = -1
        y = ts.mean(axis=0)
    else:
        ch = int(np.clip(channel, 0, n_regions - 1))
        y = ts[ch]
    return clean(
        {
            "kind": "signal_channel",
            "channel": ch,
            "name": _channel_name(ch, labels),
            "n_regions": int(n_regions),
            "t": list(range(n_timepoints)),
            "y": y,
        }
    )


def cohort_bands_spec(cohort: dict, **_) -> dict:
    """Cohort IMF-frequency histogram + the s5/s4/s* band ranges (the notebook figure).

    Reconstructs the collaborator's frequency *and* period histograms (in Hz /
    seconds, ``fs = 1/TR``): pooled per-IMF characteristic frequencies (median
    instantaneous frequency) across all subjects/ROIs, filtered to ``0 < f < 1``
    Hz, with the band ranges and per-IMF-index cluster centres for
    shading/annotation. ``scheme`` is ``canonical`` (fixed edges) or
    ``data_driven``.
    """
    import math

    freqs = np.asarray(cohort["all_freqs"], dtype=float)
    freqs = freqs[np.isfinite(freqs) & (freqs > 0) & (freqs < 1.0)]  # notebook filter
    scheme = cohort.get("scheme")
    if freqs.size < 2 or scheme is None:
        return {"kind": "cohort_bands", "error": "not enough IMF frequencies to build the histogram"}

    bands = {name: list(rng) for name, rng in scheme.bands.items()}
    # Bins ~ the notebook's ceil(log2(n) + 30) rule.
    nbins = max(8, math.ceil(math.log2(freqs.size) + 30))
    hi = max(0.2, float(np.percentile(freqs, 99.7)), bands["sstar"][1])
    counts, edges = np.histogram(freqs, bins=nbins, range=(0.0, hi))
    # Period (s) histogram — the notebook's companion panel. Bound the axis by the
    # slowest band edge (Slow-5 low), NOT a percentile of 1/f: near-DC drift IMFs
    # (f→0) blow up any period percentile to ~1e17 s, collapsing every real period
    # into the first bin (the "one bar" bug). 1.5x the Slow-5 period leaves a little
    # headroom above the slowest oscillation of interest.
    periods = 1.0 / freqs
    p_hi = 1.5 / bands["s5"][0]
    p_counts, p_edges = np.histogram(periods, bins=nbins, range=(0.0, p_hi))
    centers = {int(k): float(v) for k, v in scheme.centers.items()}
    # IMFs per index that contributed (cluster sizes), for hover/context.
    cluster_n = {int(k): int(np.size(v)) for k, v in cohort["per_index"].items()}
    return clean(
        {
            "kind": "cohort_bands",
            "contrast": cohort["contrast"],
            "processing": cohort["processing"],
            "scheme": cohort.get("scheme_name", "canonical"),
            "units": "Hz",
            "sample_rate": float(cohort.get("sample_rate", 1.0)),
            "n_subjects": int(cohort["n_subjects"]),
            "n_imf_total": int(freqs.size),
            "counts": counts,
            "edges": edges,
            "period_counts": p_counts,
            "period_edges": p_edges,
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
    labels = _names_for(len(names) or max(ch + 1, 1), names)
    name = _channel_name(ch, labels)
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
            "scheme": recon.get("scheme", "canonical"),
            "units": "Hz",
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
    labels = _names_for(len(names) or max(ch + 1, 1), names)
    name = _channel_name(ch, labels)
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
