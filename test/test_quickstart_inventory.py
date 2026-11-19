"""Tests for the ``quickstart.py`` data auto-detector.

``quickstart.inventory`` is the piece with real branching logic: it scans a data
folder for ``raw_data_<id>`` datasets, detects each one's atlas + subject count,
and decides whether its result bundles already exist (so ingest can be skipped).
``data_env`` builds the env-var overrides that repoint the backend at a data
folder living outside the repo (``--data PATH``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# quickstart.py lives at the repo root, not inside a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quickstart  # noqa: E402


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_atlas_detection() -> None:
    assert quickstart.atlas_of("schaefer100_november2025") == "Schaefer-100"
    assert quickstart.atlas_of("harvardoxford48") == "HarvardOxford-48"
    assert quickstart.atlas_of("harvard_oxford_cortical") == "HarvardOxford-48"
    assert quickstart.atlas_of("mystery_atlas_2027") == "unknown"


def test_inventory_detects_datasets_and_bundle_status(tmp_path: Path) -> None:
    data = tmp_path / "data"
    results = data / "correlation_matrices_results"

    # Dataset A: two subjects, fully bundled (global + bands + patients).
    a = data / "raw_data_schaefer100_test"
    (a / "sub-01").mkdir(parents=True)
    (a / "sub-02").mkdir(parents=True)
    for level in ("global", "bands", "patients"):
        _touch(results / "schaefer100_test" / level / "results.pkl")

    # Dataset B: one subject, no bundles yet → must be flagged for ingest.
    b = data / "raw_data_harvardoxford48"
    (b / "sub-0A").mkdir(parents=True)

    inv = quickstart.inventory(data, results)
    by_id = {d["id"]: d for d in inv}
    assert set(by_id) == {"schaefer100_test", "harvardoxford48"}

    a_row = by_id["schaefer100_test"]
    assert a_row["atlas"] == "Schaefer-100"
    assert a_row["n_subjects"] == 2
    assert a_row["has_bundles"] is True
    assert a_row["needs_ingest"] is False
    assert a_row["n_variants"] in (None, 0)  # no .ts.1D files → 0 variants (or None pre-install)

    b_row = by_id["harvardoxford48"]
    assert b_row["atlas"] == "HarvardOxford-48"
    assert b_row["n_subjects"] == 1
    assert b_row["has_bundles"] is False
    assert b_row["needs_ingest"] is True


def test_inventory_partial_bundles_still_needs_ingest(tmp_path: Path) -> None:
    data = tmp_path / "data"
    results = data / "correlation_matrices_results"
    ds = data / "raw_data_schaefer100_partial"
    (ds / "sub-01").mkdir(parents=True)
    _touch(results / "schaefer100_partial" / "global" / "results.pkl")  # only global

    (row,) = quickstart.inventory(data, results)
    assert row["has_bundles"] is True  # something is there …
    assert row["needs_ingest"] is True  # … but not all three levels


def test_inventory_empty_when_no_raw_datasets(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    assert quickstart.inventory(data, data / "correlation_matrices_results") == []


def test_data_env_points_only_at_existing_targets(tmp_path: Path) -> None:
    data = tmp_path / "shared"
    (data / "correlation_matrices_results").mkdir(parents=True)
    (data / "schaefer_2018").mkdir()
    (data / "raw_data_schaefer100_april2026").mkdir()
    # No HarvardOxford-Cortical.xml on disk → that override must be omitted.

    env = quickstart.data_env(data)
    assert env["MFB_DATA_ROOT"] == str(data)
    assert env["MFB_RESULTS_ROOT"] == str(data / "correlation_matrices_results")
    assert env["MFB_ATLAS_DIR"] == str(data / "schaefer_2018")
    assert env["MFB_RAW_DATA_ROOT"] == str(data / "raw_data_schaefer100_april2026")
    assert "MFB_HARVARD_OXFORD_XML" not in env


def test_data_env_falls_back_to_first_raw_dir(tmp_path: Path) -> None:
    data = tmp_path / "shared"
    (data / "raw_data_harvardoxford48").mkdir(parents=True)  # no april set present
    env = quickstart.data_env(data)
    assert env["MFB_RAW_DATA_ROOT"] == str(data / "raw_data_harvardoxford48")
