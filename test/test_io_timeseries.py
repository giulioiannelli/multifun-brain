"""Tests for raw ROI-timecourse I/O (``multifunbrain.io.timeseries``)."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.io import (
    SAMPLING_INTERVAL_SECONDS,
    discover_timecourses,
    load_timecourses,
    parse_timecourse_filename,
    sampling_rate,
)

# A realistic AFNI hand-off filename (shortened body, real prefixes/suffixes).
_CO2_BPF = (
    "Schaefer2018_100Parcels_17Networks_sub-00246757_ses-20260126"
    "_task-co2_run-02_space-MNI152NLin2009cAsym_desc-bpfBOLD.ts.1D"
)
_REST_OPTCOM = (
    "Schaefer2018_100Parcels_17Networks_sub-00246757_ses-20260126"
    "_task-rest_run-02_PCA-aic_GS-mir_space-MNI152NLin2009cAsym_desc-optcom_bold.ts.1D"
)
_REST_OPTCOM_DENOISED = (
    "Schaefer2018_100Parcels_17Networks_sub-00246757_ses-20260126"
    "_task-rest_run-02_PCA-aic_GS-mir_space-MNI152NLin2009cAsym"
    "_desc-optcomMIRDenoised_bold.ts.1D"
)


def test_parse_filename_extracts_metadata():
    meta = parse_timecourse_filename(_CO2_BPF)
    assert meta == {
        "subject": "sub-00246757",
        "session": "20260126",
        "contrast": "co2",
        "run": "02",
        "processing": "bpfBOLD",
        "atlas": "schaefer100",
    }


def test_parse_filename_disambiguates_optcom_variants():
    # The shared "optcom" prefix must not shadow the longer denoised variant.
    assert parse_timecourse_filename(_REST_OPTCOM)["processing"] == "optcom_bold"
    assert (
        parse_timecourse_filename(_REST_OPTCOM_DENOISED)["processing"]
        == "optcomMIRDenoised_bold"
    )


def test_parse_filename_rejects_non_matching():
    assert parse_timecourse_filename("not_a_timecourse.txt") is None
    assert parse_timecourse_filename("sub-1_task-foo_desc-bpfBOLD.ts.1D") is None


def test_load_timecourses_orientation(tmp_path):
    # 5 timepoints x 3 regions on disk; region-major load transposes to (3, 5).
    data = np.arange(15, dtype=float).reshape(5, 3)
    f = tmp_path / "sub-x_ses-1_task-co2_run-01_desc-bpfBOLD.ts.1D"
    np.savetxt(f, data)

    region_major = load_timecourses(f)
    assert region_major.shape == (3, 5)
    np.testing.assert_allclose(region_major, data.T)

    native = load_timecourses(f, region_major=False)
    assert native.shape == (5, 3)
    np.testing.assert_allclose(native, data)


def test_load_timecourses_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_timecourses(tmp_path / "nope.ts.1D")


def test_discover_timecourses(tmp_path):
    sub = tmp_path / "sub-00246757"
    sub.mkdir()
    arr = np.zeros((4, 3))
    for contrast in ("co2", "rest"):
        for proc in ("bpfBOLD", "optcom_bold"):
            name = (
                f"Schaefer_sub-00246757_ses-20260126_task-{contrast}"
                f"_run-01_space-MNI_desc-{proc}.ts.1D"
            )
            np.savetxt(sub / name, arr)
    # A junk file that should be ignored.
    (sub / "readme.txt").write_text("ignore me")

    found = discover_timecourses(tmp_path)
    assert len(found) == 4
    assert {e.contrast for e in found} == {"co2", "rest"}
    assert {e.processing for e in found} == {"bpfBOLD", "optcom_bold"}
    assert all(e.subject == "sub-00246757" for e in found)

    only_co2 = discover_timecourses(tmp_path, contrasts=["co2"])
    assert {e.contrast for e in only_co2} == {"co2"}


def test_discover_missing_root(tmp_path):
    assert discover_timecourses(tmp_path / "absent") == []


def test_sampling_rate_per_variant():
    # fs = 1/TR, with the per-variant TR from the April hand-off.
    assert sampling_rate("bpfBOLD") == pytest.approx(1.0 / 1.353)
    assert sampling_rate("bpfVASO") == pytest.approx(1.0 / 1.353)
    assert sampling_rate("optcom_bold") == pytest.approx(1.0 / 0.98)
    assert sampling_rate("optcomMIRDenoised_bold") == pytest.approx(1.0 / 0.98)
    assert sampling_rate("MIRNoise_bold") == pytest.approx(1.0 / 0.98)
    # Unknown variant degrades to cycles/sample (fs = 1.0).
    assert sampling_rate("nonexistent") == 1.0
    # The bpf variants are sampled slower than the optcom/MIR family.
    assert SAMPLING_INTERVAL_SECONDS["bpfBOLD"] > SAMPLING_INTERVAL_SECONDS["optcom_bold"]


def test_parse_filename_kw_scheme():
    # Older scheme: no subject/task/run in the name; modality token is processing.
    s = parse_timecourse_filename("Schaefer2018_100Parcels_17Networks_clean_kwfurN_Bold.ts.1D")
    assert s == {
        "subject": None,
        "session": None,
        "contrast": None,
        "run": None,
        "processing": "clean_kwfurN_Bold",
        "atlas": "schaefer100",
    }
    ho = parse_timecourse_filename("HarvardOxford_48Parcels_kwCBF4D.ts.1D")
    assert ho["processing"] == "kwCBF4D"
    assert ho["atlas"] == "harvardoxford48"
    assert ho["contrast"] is None


def test_discover_kw_subject_from_parent_and_skips_discarded(tmp_path):
    sub = tmp_path / "sub-00187189"
    sub.mkdir()
    arr = np.zeros((4, 3))
    for tok in ("kwCBF4D", "kwfurN_Bold"):
        np.savetxt(sub / f"HarvardOxford_48Parcels_{tok}.ts.1D", arr)
    # A `discarded/` subtree must be ignored entirely.
    disc = tmp_path / "discarded" / "sub-99999999"
    disc.mkdir(parents=True)
    np.savetxt(disc / "HarvardOxford_48Parcels_kwCBF4D.ts.1D", arr)

    found = discover_timecourses(tmp_path)
    assert len(found) == 2  # discarded one excluded
    assert {e.subject for e in found} == {"sub-00187189"}  # from parent dir
    assert {e.processing for e in found} == {"kwCBF4D", "kwfurN_Bold"}
    assert all(e.contrast is None and e.run is None for e in found)
    assert all(e.atlas == "harvardoxford48" for e in found)
