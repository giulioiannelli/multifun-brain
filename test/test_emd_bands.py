"""Tests for EMD frequency-band analysis (``multifunbrain.analysis.emd_bands``)."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.analysis.emd_bands import (
    BAND_ORDER,
    assign_imfs,
    estimate_band_edges,
    reconstruct_bands,
    sift_with_frequencies,
)

emd = pytest.importorskip("emd")


def _three_tone(n=2000, seed=0):
    """Signal with three well-separated tones (cycles/sample: 0.20, 0.05, 0.0125)."""
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    x = (
        1.0 * np.sin(2 * np.pi * 0.20 * t)
        + 1.0 * np.sin(2 * np.pi * 0.05 * t)
        + 1.0 * np.sin(2 * np.pi * 0.0125 * t)
    )
    return x + 0.02 * rng.standard_normal(n)


def test_sift_with_frequencies_recovers_tones():
    imfs, freqs = sift_with_frequencies(_three_tone())
    assert imfs.ndim == 2
    assert freqs.shape[0] == imfs.shape[1]
    finite = freqs[np.isfinite(freqs)]
    # The three injected tones should each be near some IMF's characteristic freq.
    for tone in (0.20, 0.05, 0.0125):
        assert np.min(np.abs(finite - tone)) < 0.02, (tone, finite)


def test_sift_reconstructs_input():
    x = _three_tone()
    imfs, _ = sift_with_frequencies(x)
    np.testing.assert_allclose(imfs.sum(axis=1), x, atol=1e-6)


def test_estimate_band_edges_orders_low_to_high():
    # Four clusters high→low; top 3 become sstar/s4/s5, the 4th defines drift_max.
    per_index = {
        0: np.full(50, 0.20),
        1: np.full(50, 0.05),
        2: np.full(50, 0.0125),
        3: np.full(50, 0.003),
    }
    scheme = estimate_band_edges(per_index)
    assert set(scheme.bands) == set(BAND_ORDER)
    s5 = scheme.bands["s5"]
    s4 = scheme.bands["s4"]
    star = scheme.bands["sstar"]
    # Non-overlapping, monotonic: s5 < s4 < sstar.
    assert s5[0] < s5[1] <= s4[0] < s4[1] <= star[0] < star[1]
    # Each cluster centre sits inside its band.
    assert s5[0] <= 0.0125 < s5[1]
    assert s4[0] <= 0.05 < s4[1]
    assert star[0] <= 0.20 < star[1]
    assert scheme.drift_max == pytest.approx(s5[0])


def test_estimate_band_edges_drops_degenerate_clusters():
    per_index = {
        0: np.full(50, 0.20),
        1: np.full(50, 0.05),
        2: np.full(50, 0.0125),
        3: np.full(50, 0.003),
        4: np.array([0.001, 0.001]),  # too few samples -> dropped
    }
    scheme = estimate_band_edges(per_index, min_count=5)
    assert 4 not in scheme.centers


def test_assign_and_reconstruct_bands():
    x = _three_tone()
    imfs, freqs = sift_with_frequencies(x)
    # Bands straddling the three tones.
    bands = {"s5": (0.008, 0.025), "s4": (0.025, 0.10), "sstar": (0.10, 0.35)}
    assignment = assign_imfs(freqs, bands)
    # Every band should capture at least one IMF for this clean signal.
    assert all(assignment[b] for b in BAND_ORDER)

    signals, assignment2 = reconstruct_bands(imfs, freqs, bands)
    assert assignment2 == assignment
    for name in BAND_ORDER:
        assert signals[name].shape == x.shape
    # The summed band signals should not exceed the full reconstruction in energy.
    total = sum(signals[b] for b in BAND_ORDER)
    assert np.var(total) <= np.var(x) + 1e-6
