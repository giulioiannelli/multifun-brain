"""Tests for the dashboard results catalog (``dashboard.backend.catalog``).

Covers label parsing for both the April (co2/rest) and contrast-less ``kw`` label
shapes, and the ``list_datasets`` filter that keeps only bundles under a known raw
dataset id.
"""

from __future__ import annotations

import pickle

import pytest

from dashboard.backend import catalog
from multifunbrain import PipelineConfig, PipelineResult


@pytest.mark.parametrize(
    "label,expected",
    [
        # April: task = co2/rest, contrast = modality, processing = pipeline.
        ("global/co2_MIRNoise_bold",
         {"level": "global", "task": "co2", "contrast": "bold", "processing": "MIRnoise",
          "band": None, "subject": None}),
        ("global/rest_bpfBOLD",
         {"task": "rest", "contrast": "bold", "processing": "bpf", "band": None}),
        ("band/co2_bpfVASO/s4",
         {"level": "band", "task": "co2", "contrast": "vaso", "processing": "bpf", "band": "s4"}),
        ("patient/sub-02/co2_bpfVASO/sstar",
         {"level": "patient", "subject": "sub-02", "task": "co2", "contrast": "vaso",
          "processing": "bpf", "band": "sstar"}),
        # kw: no task; the modality is parsed out of the token as the contrast.
        ("global/kwoptcomMIRDenoised_bold",
         {"task": None, "contrast": "bold", "processing": "optcomMIRdenoised"}),
        ("global/clean_kwCBF4D",
         {"task": None, "contrast": "cbf", "processing": "clean"}),
        ("band/kwBOLD4D/s4",
         {"task": None, "contrast": "bold", "processing": "raw", "band": "s4"}),
        ("patient/sub-01/kwfurN_Bold",
         {"subject": "sub-01", "task": None, "contrast": "bold", "processing": "furN", "band": None}),
        ("patient/sub-01/kwfurN_Bold/s5",
         {"subject": "sub-01", "task": None, "contrast": "bold", "processing": "furN", "band": "s5"}),
    ],
)
def test_parse_label(label, expected):
    got = catalog.parse_label(label)
    for key, value in expected.items():
        assert got[key] == value, (label, key, got[key], value)


def test_parse_label_empty():
    out = catalog.parse_label("")
    assert all(v is None for v in out.values())


def _write_bundle(directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    result = PipelineResult(config=PipelineConfig(), label=label)
    with open(directory / "results.pkl", "wb") as f:
        pickle.dump([result], f)


def test_list_datasets_filters_to_known_ids(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _write_bundle(root / "schaefer100_november2025" / "global", "global/kwBOLD4D")
    _write_bundle(root / "legacy_junk" / "global", "global/whatever")
    monkeypatch.setattr(catalog.config, "RESULTS_ROOT", root)
    monkeypatch.setattr(catalog, "_known_dataset_ids", lambda: {"schaefer100_november2025"})

    ids = {d["id"] for d in catalog.list_datasets()}
    assert "schaefer100_november2025/global" in ids
    assert "legacy_junk/global" not in ids


def test_list_datasets_no_filter_when_no_raw(tmp_path, monkeypatch):
    # A results-only checkout (no raw datasets present) must not hide everything.
    root = tmp_path / "results"
    _write_bundle(root / "anything" / "global", "global/x")
    monkeypatch.setattr(catalog.config, "RESULTS_ROOT", root)
    monkeypatch.setattr(catalog, "_known_dataset_ids", lambda: set())

    ids = {d["id"] for d in catalog.list_datasets()}
    assert "anything/global" in ids


def test_list_datasets_skips_root_bundle(tmp_path, monkeypatch):
    root = tmp_path / "results"
    _write_bundle(root, "global/loose")  # bundle sitting at the root itself
    _write_bundle(root / "harvardoxford48" / "global", "global/kwCBF4D")
    monkeypatch.setattr(catalog.config, "RESULTS_ROOT", root)
    monkeypatch.setattr(catalog, "_known_dataset_ids", lambda: {"harvardoxford48"})

    ids = {d["id"] for d in catalog.list_datasets()}
    assert ids == {"harvardoxford48/global"}  # the root "." bundle is excluded
