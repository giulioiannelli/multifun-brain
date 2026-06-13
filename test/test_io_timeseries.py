"""Tests for raw ROI-timecourse I/O (``multifunbrain.io.timeseries``)."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.io import (
    discover_timecourses,
    load_timecourses,
    parse_timecourse_filename,
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
