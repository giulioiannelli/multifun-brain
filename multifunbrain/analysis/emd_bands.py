"""EMD-based frequency-band analysis for ROI timecourses.

Empirical Mode Decomposition (EMD) splits a signal into Intrinsic Mode Functions
(IMFs) ordered fast → slow. Each IMF has a *characteristic frequency* — by
default the **median instantaneous frequency** from its Hilbert transform (the
HHT estimator the collaborator's notebook uses); an amplitude-weighted mean is
available via ``method="weighted_mean"``.

An IMF is assigned to the ``s5`` / ``s4`` / ``sstar`` band whose range contains
its characteristic frequency, and a per-band signal is reconstructed by summing
the IMFs in that band (the same band signals the per-band correlation matrices
are built from). Two band schemes are provided:

- :data:`CANONICAL_BANDS` / :func:`canonical_scheme` — **fixed** resting-state
  slow-oscillation edges (Hz), the default and what the collaborator uses:
  Slow-5 ``[0.010, 0.027]`` < Slow-4 ``[0.027, 0.073]`` < S* ``[0.073, 0.180]``.
- :func:`estimate_band_edges` — **data-driven** edges: geometric midpoints
  between the per-IMF-index frequency clusters (an exploratory alternative).

Frequencies are in **Hz** when ``sample_rate = 1/TR`` is passed, or in cycles per
sample when ``sample_rate=1.0``. The canonical bands are defined in Hz, so the
real per-variant TR must be used to compare against them. Bands are ordered
low → high frequency: Slow-5 < Slow-4 < S*.

``emd`` is an optional dependency (declared in the dashboard extra); it is
imported lazily so importing this module never requires it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BAND_LABELS",
    "BAND_ORDER",
    "CANONICAL_BANDS",
    "BandScheme",
    "assign_imfs",
    "canonical_scheme",
    "estimate_band_edges",
    "reconstruct_bands",
    "sift_with_frequencies",
]

# Bands ordered low → high frequency (Slow-5 is the slowest).
BAND_ORDER = ("s5", "s4", "sstar")
BAND_LABELS = {"s5": "Slow-5", "s4": "Slow-4", "sstar": "S*"}

# Canonical resting-state slow-oscillation bands (Hz) — Buzsáki/Zuo classes, and
# the fixed edges the collaborator's notebook classifies IMFs into. Contiguous:
# an IMF on a shared edge falls in the lower band (see :func:`assign_imfs`).
CANONICAL_BANDS = {
    "s5": (0.010, 0.027),
    "s4": (0.027, 0.073),
    "sstar": (0.073, 0.180),
}


def sift_with_frequencies(
    signal, sample_rate: float = 1.0, method: str = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """EMD-sift *signal* and return ``(imfs, freqs)``.

    ``imfs`` has shape ``(n_samples, n_imf)`` (so ``imfs.sum(axis=1)`` reconstructs
    the input). ``freqs`` has shape ``(n_imf,)`` — each IMF's characteristic
    frequency (Hz when ``sample_rate = 1/TR``; cycles/sample when ``=1``). An IMF
    with no usable instantaneous frequency gets ``nan``.

    Parameters
    ----------
    method : {"median", "weighted_mean"}, default "median"
        ``"median"`` — median of the Hilbert instantaneous frequency (the HHT
        estimator the collaborator's notebook uses). ``"weighted_mean"`` —
        amplitude-weighted mean ``Σ(IF·IA)/ΣIA``.
    """
    import emd

    x = np.asarray(signal, dtype=float)
    imfs = np.asarray(emd.sift.sift(x))
    if imfs.ndim != 2 or imfs.shape[1] == 0:
        imfs = x.reshape(-1, 1)
    _, instf, insta = emd.spectra.frequency_transform(imfs, sample_rate, "hilbert")
    instf = np.asarray(instf, dtype=float)
    insta = np.asarray(insta, dtype=float)
    instf = np.where(np.isfinite(instf), instf, np.nan)
    if method == "weighted_mean":
        weight = insta.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freqs = (instf * insta).sum(axis=0) / weight
        freqs = np.where(weight > 0, freqs, np.nan)
    else:  # "median" — collaborator's HHT characteristic frequency
        with warnings.catch_warnings():  # all-NaN IMF column -> nan (benign)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            freqs = np.nanmedian(instf, axis=0)
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


def canonical_scheme(centers: dict | None = None) -> BandScheme:
    """The fixed-edge :class:`BandScheme` (the collaborator's default), bands in Hz.

    ``centers`` (per-IMF-index cluster medians) is optional context for display;
    it does not affect the fixed :data:`CANONICAL_BANDS` edges.
    """
    return BandScheme(
        bands=dict(CANONICAL_BANDS),
        centers=dict(centers) if centers else {},
        drift_max=CANONICAL_BANDS["s5"][0],
    )


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

    Returns ``{band_name: [imf_index, ...]}``. Bands are tested low → high
    (``s5`` → ``s4`` → ``sstar``) with **closed** intervals ``[lo, hi]`` and the
    first match wins, so for the contiguous canonical edges an IMF exactly on a
    shared boundary lands in the lower band — exactly the collaborator's nested
    ``if 0.010<=f<=0.027 … elif 0.027<f<=0.073 …`` classification. IMFs that fall
    in no band (slow drift below ``s5``, or above the top edge) are omitted.
    """
    out: dict[str, list[int]] = {name: [] for name in BAND_ORDER}
    for i, f in enumerate(np.asarray(freqs, dtype=float)):
        if not np.isfinite(f):
            continue
        for name in BAND_ORDER:
            lo, hi = bands[name]
            if lo <= f <= hi:
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
