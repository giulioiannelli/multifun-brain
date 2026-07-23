"""Tests for the Signal-tab cross-subject average (``dashboard.backend.timeseries``).

``subject = AVG_SUBJECT`` averages every subject's raw timecourse for a variant
element-wise; carpet / single-channel / EMD / band views all run on that mean.
"""

from __future__ import annotations

import numpy as np

from dashboard.backend import timeseries

_NAME = "HarvardOxford_48Parcels_kwCBF4D.ts.1D"


def test_get_timecourses_average(tmp_path, monkeypatch):
    ds = tmp_path / "raw_data_test"
    for sub, value in [("sub-01", 1.0), ("sub-02", 3.0)]:
        d = ds / sub
        d.mkdir(parents=True)
        np.savetxt(d / _NAME, np.full((5, 48), value))  # (timepoints, regions) on disk
    monkeypatch.setattr(timeseries, "_root_for", lambda dataset=None: ds)

    # Single subject: region-major (48, 5), constant at its value.
    one = timeseries.get_timecourses("test", "sub-01", "", "kwCBF4D")
    assert one is not None and one.shape == (48, 5)
    np.testing.assert_allclose(one, 1.0)

    # Average: element-wise mean across the two subjects.
    avg = timeseries.get_timecourses("test", timeseries.AVG_SUBJECT, "", "kwCBF4D")
    assert avg is not None and avg.shape == (48, 5)
    np.testing.assert_allclose(avg, 2.0)


def test_average_truncates_mismatched_lengths(tmp_path, monkeypatch):
    ds = tmp_path / "raw_data_test"
    for sub, n_t in [("sub-01", 6), ("sub-02", 4)]:
        d = ds / sub
        d.mkdir(parents=True)
        np.savetxt(d / _NAME, np.ones((n_t, 48)))
    monkeypatch.setattr(timeseries, "_root_for", lambda dataset=None: ds)
    avg = timeseries.get_timecourses("test", timeseries.AVG_SUBJECT, "", "kwCBF4D")
    assert avg is not None and avg.shape == (48, 4)  # truncated to the shortest run


def test_sift_channel_average_runs(tmp_path, monkeypatch):
    ds = tmp_path / "raw_data_test"
    t = np.linspace(0.0, 8.0 * np.pi, 128)
    base = np.sin(t)
    rng = np.random.default_rng(0)
    for sub in ("sub-01", "sub-02", "sub-03"):
        d = ds / sub
        d.mkdir(parents=True)
        data = base[:, None] + 0.1 * rng.normal(size=(128, 48))  # (timepoints, regions)
        np.savetxt(d / _NAME, data)
    monkeypatch.setattr(timeseries, "_root_for", lambda dataset=None: ds)

    out = timeseries.sift_channel("test", timeseries.AVG_SUBJECT, "", "kwCBF4D", channel=0)
    assert out is not None
    assert out["channel"] == 0
    assert out["signal"].shape == (128,)
    assert out["imfs"].shape[1] == 128  # (n_imf, n_timepoints)
    assert out["n_imf"] >= 1


def _write_oscillatory(ds, subjects=("sub-01", "sub-02", "sub-03"), n_t=128, n_r=48):
    """Multi-subject synthetic set: a shared slow+fast oscillation + per-ROI noise."""
    t = np.linspace(0.0, 8.0 * np.pi, n_t)
    base = np.sin(t) + 0.4 * np.sin(4.0 * t)
    rng = np.random.default_rng(0)
    for sub in subjects:
        d = ds / sub
        d.mkdir(parents=True)
        data = base[:, None] + 0.1 * rng.normal(size=(n_t, n_r))  # (timepoints, regions)
        np.savetxt(d / _NAME, data)


def test_band_carpet_shapes_and_contract(tmp_path, monkeypatch):
    ds = tmp_path / "raw_data_test"
    _write_oscillatory(ds)
    monkeypatch.setattr(timeseries, "_root_for", lambda dataset=None: ds)

    # Each canonical band → a region-major band-reconstructed carpet, same shape.
    for band in ("s5", "s4", "sstar"):
        carpet = timeseries.band_carpet("test", "sub-01", "", "kwCBF4D", band)
        assert carpet is not None and carpet.shape == (48, 128)
        assert np.isfinite(carpet).all()

    # "full" is not served here (the route uses get_timecourses); unknown → None.
    assert timeseries.band_carpet("test", "sub-01", "", "kwCBF4D", "full") is None
    assert timeseries.band_carpet("test", "sub-01", "", "kwCBF4D", "nope") is None

    # The cross-subject average is band-reconstructed too.
    avg = timeseries.band_carpet("test", timeseries.AVG_SUBJECT, "", "kwCBF4D", "s4")
    assert avg is not None and avg.shape == (48, 128)


def test_cohort_bands_subject_filter(tmp_path, monkeypatch):
    ds = tmp_path / "raw_data_test"
    _write_oscillatory(ds)
    monkeypatch.setattr(timeseries, "_root_for", lambda dataset=None: ds)

    whole = timeseries.cohort_bands("test", "", "kwCBF4D", "canonical")
    assert whole is not None and whole["n_subjects"] == 3 and whole["subject"] is None

    one = timeseries.cohort_bands("test", "", "kwCBF4D", "canonical", "sub-01")
    assert one is not None and one["n_subjects"] == 1 and one["subject"] == "sub-01"
    # A single subject pools strictly fewer IMF frequencies than the whole cohort.
    assert one["all_freqs"].size < whole["all_freqs"].size

    # An unknown subject yields no pool (404 at the route).
    assert timeseries.cohort_bands("test", "", "kwCBF4D", "canonical", "sub-99") is None
