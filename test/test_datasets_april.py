"""Tests for the April-batch metadata-aware loader.

All tests are data-independent: they exercise the directory walker and
filename parsers against synthetic trees in ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multifunbrain.datasets.april import (
    BANDS,
    CONTRASTS,
    LEVELS,
    N_PARCELS,
    N_TIMEPOINTS_PER_PROCESSING,
    PROCESSING_VARIANTS,
    AprilEntry,
    discover_april,
    entries_to_dataframe,
    gamma_for,
    label_for,
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_synthetic_april(root: Path) -> None:
    """Create a tiny synthetic April-style tree with one entry per level."""
    glob_dir = root / "freq-contrast-global" / "co2_bpfBOLD"
    glob_dir.mkdir(parents=True)
    (glob_dir / "co2_bpfBOLD_GLOBAL.pkl").touch()

    band_dir = root / "freq-contrast-inter" / "rest_optcom_bold"
    band_dir.mkdir(parents=True)
    for band in ("s4", "s5", "sstar"):
        (band_dir / f"rest_optcom_bold_{band}.pkl").touch()

    pat_dir = root / "freq-user-constrast-inter" / "sub-00246757" / "co2_bpfBOLD"
    pat_dir.mkdir(parents=True)
    (pat_dir / "sub-00246757_co2_bpfBOLD_s4.pkl").touch()

    # Decoys / excluded files
    (root / "freq-contrast-global" / ".DS_Store").touch()
    (root / "All_Emd_diz.pkl").touch()
    hist_dir = root / "hist_C_I"
    hist_dir.mkdir(parents=True)
    (hist_dir / "co2_bpfBOLD.png").touch()
    # A pkl in hist_C_I should still be excluded (defensive)
    (hist_dir / "co2_bpfBOLD.pkl").touch()


# ──────────────────────────────────────────────────────────────────────
# Filename parsing & label generation
# ──────────────────────────────────────────────────────────────────────


def test_constants_consistent() -> None:
    assert CONTRASTS == ("co2", "rest")
    assert "bpfBOLD" in PROCESSING_VARIANTS
    assert "optcomMIRDenoised_bold" in PROCESSING_VARIANTS
    assert BANDS == ("s4", "s5", "sstar")
    assert LEVELS == ("global", "band", "patient")


def test_label_for_global() -> None:
    e = AprilEntry(
        level="global",
        contrast="co2",
        processing="bpfBOLD",
        band=None,
        subject=None,
        path=Path("dummy.pkl"),
    )
    assert label_for(e) == "global/co2_bpfBOLD"


def test_label_for_band() -> None:
    e = AprilEntry(
        level="band",
        contrast="rest",
        processing="optcom_bold",
        band="s5",
        subject=None,
        path=Path("dummy.pkl"),
    )
    assert label_for(e) == "band/rest_optcom_bold/s5"


def test_label_for_patient() -> None:
    e = AprilEntry(
        level="patient",
        contrast="co2",
        processing="bpfBOLD",
        band="s4",
        subject="sub-00246757",
        path=Path("dummy.pkl"),
    )
    assert label_for(e) == "patient/sub-00246757/co2_bpfBOLD/s4"


# ──────────────────────────────────────────────────────────────────────
# discover_april
# ──────────────────────────────────────────────────────────────────────


def test_discover_april_missing_root(tmp_path: Path) -> None:
    assert discover_april(tmp_path / "does-not-exist") == []


def test_discover_april_empty(tmp_path: Path) -> None:
    assert discover_april(tmp_path) == []


def test_discover_april_full_synthetic(tmp_path: Path) -> None:
    _make_synthetic_april(tmp_path)
    entries = discover_april(tmp_path)
    # 1 global + 3 band (s4, s5, sstar) + 1 patient = 5
    assert len(entries) == 5
    levels = {e.level for e in entries}
    assert levels == {"global", "band", "patient"}

    # Excluded files
    paths = [str(e.path) for e in entries]
    assert all(".DS_Store" not in p for p in paths)
    assert all("All_Emd_diz" not in p for p in paths)
    assert all("hist_C_I" not in p for p in paths)


def test_discover_april_filter_by_level(tmp_path: Path) -> None:
    _make_synthetic_april(tmp_path)
    only_global = discover_april(tmp_path, levels=["global"])
    assert len(only_global) == 1
    assert only_global[0].level == "global"

    only_band = discover_april(tmp_path, levels=["band"])
    assert len(only_band) == 3
    assert {e.band for e in only_band} == {"s4", "s5", "sstar"}


def test_discover_april_invalid_level_raises(tmp_path: Path) -> None:
    _make_synthetic_april(tmp_path)
    with pytest.raises(ValueError, match="Unknown level"):
        discover_april(tmp_path, levels=["nonsense"])  # type: ignore[list-item]


def test_discover_april_sorted_deterministically(tmp_path: Path) -> None:
    _make_synthetic_april(tmp_path)
    entries = discover_april(tmp_path)
    keys = [
        (e.level, e.contrast, e.processing, e.band or "", e.subject or "")
        for e in entries
    ]
    assert keys == sorted(keys)


# ──────────────────────────────────────────────────────────────────────
# entries_to_dataframe
# ──────────────────────────────────────────────────────────────────────


def test_entries_to_dataframe_schema(tmp_path: Path) -> None:
    _make_synthetic_april(tmp_path)
    entries = discover_april(tmp_path)
    df = entries_to_dataframe(entries)
    expected_cols = {"level", "contrast", "processing", "band", "subject", "path", "label"}
    assert set(df.columns) == expected_cols
    assert len(df) == len(entries)
    assert isinstance(df, pd.DataFrame)


def test_entries_to_dataframe_empty() -> None:
    df = entries_to_dataframe([])
    assert len(df) == 0
    assert set(df.columns) == {"level", "contrast", "processing", "band", "subject", "path", "label"}


# ──────────────────────────────────────────────────────────────────────
# gamma_for
# ──────────────────────────────────────────────────────────────────────


def test_gamma_for_known_processing() -> None:
    entry = AprilEntry(
        level="global",
        contrast="co2",
        processing="bpfBOLD",
        band=None,
        subject=None,
        path=Path("dummy.pkl"),
    )
    gamma = gamma_for(entry)
    assert gamma is not None
    assert gamma == pytest.approx(N_PARCELS / 442)


def test_gamma_for_optcom_variants() -> None:
    for proc in ("optcom_bold", "optcomMIRDenoised_bold", "MIRNoise_bold"):
        entry = AprilEntry(
            level="global",
            contrast="rest",
            processing=proc,
            band=None,
            subject=None,
            path=Path("dummy.pkl"),
        )
        gamma = gamma_for(entry)
        assert gamma is not None
        assert gamma == pytest.approx(N_PARCELS / 597)


def test_gamma_for_unknown_processing_returns_none() -> None:
    entry = AprilEntry(
        level="global",
        contrast="co2",
        processing="not-a-real-variant",
        band=None,
        subject=None,
        path=Path("dummy.pkl"),
    )
    assert gamma_for(entry) is None


def test_n_timepoints_table_covers_all_documented_variants() -> None:
    for proc in PROCESSING_VARIANTS:
        assert proc in N_TIMEPOINTS_PER_PROCESSING, (
            f"PROCESSING_VARIANTS contains {proc!r} but no n_timepoints lookup"
        )
