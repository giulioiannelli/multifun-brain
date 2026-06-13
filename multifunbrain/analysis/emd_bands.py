"""EMD-based frequency-band analysis for ROI timecourses.

Empirical Mode Decomposition (EMD) splits a signal into Intrinsic Mode Functions
(IMFs) ordered fast → slow. Each IMF has a *characteristic frequency* — the
amplitude-weighted mean instantaneous frequency from its Hilbert transform.
Pooled across ROIs/subjects, the IMFs cluster tightly by index into
well-separated frequency clusters, and the **data-driven band edges** are simply
the geometric midpoints between consecutive cluster centres. This is how the
``s5`` / ``s4`` / ``sstar`` bands are defined here, and how a per-band signal is
reconstructed: sum the IMFs whose characteristic frequency falls inside the band
(the same band signals that the per-band correlation matrices are built from).

Frequencies are in **cycles per sample** when ``sample_rate=1.0`` — multiply by
the sampling rate ``1/TR`` to get Hz. Bands are ordered low → high frequency:
Slow-5 < Slow-4 < S*.

``emd`` is an optional dependency (declared in the dashboard extra); it is
imported lazily so importing this module never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BAND_LABELS",
    "BAND_ORDER",
    "BandScheme",
    "assign_imfs",
    "estimate_band_edges",
    "reconstruct_bands",
    "sift_with_frequencies",
]

# Bands ordered low → high frequency (Slow-5 is the slowest).
BAND_ORDER = ("s5", "s4", "sstar")
BAND_LABELS = {"s5": "Slow-5", "s4": "Slow-4", "sstar": "S*"}


def sift_with_frequencies(
    signal, sample_rate: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """EMD-sift *signal* and return ``(imfs, freqs)``.

    ``imfs`` has shape ``(n_samples, n_imf)`` (so ``imfs.sum(axis=1)`` reconstructs
    the input). ``freqs`` has shape ``(n_imf,)`` — the amplitude-weighted mean
    instantaneous frequency of each IMF (cycles/sample when ``sample_rate=1``);
    an IMF with no amplitude gets ``nan``.
    """
    import emd

    x = np.asarray(signal, dtype=float)
    imfs = np.asarray(emd.sift.sift(x))
    if imfs.ndim != 2 or imfs.shape[1] == 0:
        imfs = x.reshape(-1, 1)
    _, instf, insta = emd.spectra.frequency_transform(imfs, sample_rate, "hilbert")
    instf = np.asarray(instf, dtype=float)
    insta = np.asarray(insta, dtype=float)
    weight = insta.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        freqs = (instf * insta).sum(axis=0) / weight
    freqs = np.where(weight > 0, freqs, np.nan)
    return imfs, freqs


@dataclass(frozen=True)
class BandScheme:
    """Data-driven band edges + the IMF clusters they came from.

    ``bands`` maps each band name to its ``(lo, hi)`` frequency range
    (cycles/sample). ``centers`` is the per-IMF-index cluster centre
    (median characteristic frequency). ``drift_max`` is the lower edge of the
    slowest band — IMFs below it are ultra-slow drift, excluded from every band.
    """

    bands: dict
    centers: dict
    drift_max: float


def _geom_mean(a: float, b: float) -> float:
    return float(np.sqrt(a * b))


def estimate_band_edges(
    per_index_freqs: dict, n_bands: int = 3, min_count: int = 5
) -> BandScheme:
    """Estimate band edges from per-IMF-index frequency clusters.

    Parameters
    ----------
    per_index_freqs : dict[int, array-like]
        Characteristic frequencies pooled across the cohort, keyed by IMF index.
    n_bands : int, default 3
        Number of bands (the ``n_bands`` highest-frequency IMF clusters become,
        high → low, ``sstar`` / ``s4`` / ``s5``).
    min_count : int, default 5
        Drop IMF indices with fewer observations than this (degenerate clusters).

    Returns
    -------
    BandScheme
        ``bands`` (name → ``(lo, hi)``), ``centers`` (index → median freq),
        ``drift_max`` (lower edge of the slowest band).
    """
    centers = {}
    for k, vals in per_index_freqs.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size >= min_count:
            centers[int(k)] = float(np.median(arr))

    if not centers:
        raise ValueError("no usable IMF clusters to estimate bands from")

    # Cluster centres sorted high → low frequency.
    ordered = sorted(centers.values(), reverse=True)
    names_hi_to_lo = list(reversed(BAND_ORDER))[:n_bands]  # ['sstar','s4','s5']
    n = len(names_hi_to_lo)

    # Pad with geometrically-spaced centres if the cohort yielded fewer clusters.
    while len(ordered) < n + 1:
        ordered.append(ordered[-1] / 2.0)

    top = ordered[:n]
    # Internal edges = geometric midpoints between consecutive top clusters.
    inner = [_geom_mean(top[i], top[i + 1]) for i in range(n - 1)]
    drift_max = _geom_mean(top[-1], ordered[n])  # below this = drift
    upper = min(_geom_mean(top[0], top[0] * (top[0] / top[1])) if n > 1 else top[0] * 1.5, 0.5)

    bounds = [upper, *inner, drift_max]  # high → low
    bands = {}
    for i, name in enumerate(names_hi_to_lo):
        bands[name] = (float(bounds[i + 1]), float(bounds[i]))
    return BandScheme(bands=bands, centers=centers, drift_max=float(drift_max))


def assign_imfs(freqs, bands: dict) -> dict:
    """Assign each IMF index to a band by its characteristic frequency.

    Returns ``{band_name: [imf_index, ...]}``. IMFs that fall in no band (drift /
    above the top edge) are simply omitted.
    """
    out: dict[str, list[int]] = {name: [] for name in BAND_ORDER}
    for i, f in enumerate(np.asarray(freqs, dtype=float)):
        if not np.isfinite(f):
            continue
        for name in BAND_ORDER:
            lo, hi = bands[name]
            if lo <= f < hi:
                out[name].append(i)
                break
    return out


def reconstruct_bands(imfs, freqs, bands: dict) -> tuple[dict, dict]:
    """Reconstruct one signal per band by summing its IMFs.

    Returns ``(signals, assignment)`` where ``signals[name]`` is the summed
    timecourse (zeros if no IMF falls in the band) and ``assignment`` is the
    band → IMF-index map from :func:`assign_imfs`.
    """
    imfs = np.asarray(imfs, dtype=float)
    n_samples = imfs.shape[0]
    assignment = assign_imfs(freqs, bands)
    signals = {}
    for name in BAND_ORDER:
        idx = assignment[name]
        signals[name] = imfs[:, idx].sum(axis=1) if idx else np.zeros(n_samples)
    return signals, assignment
