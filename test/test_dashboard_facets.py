"""Tests for the shared variant-token facet parser (``dashboard.backend.facets``).

Locks the token -> (task, contrast, processing) mapping for the real April + kw
variant tokens, and asserts the derived ``processing`` values stay **distinct
within each (task, contrast) group** — the selector needs a unique processing per
(dataset, task, contrast) so no two results collapse onto one dropdown entry.
"""

from __future__ import annotations

import pytest

from dashboard.backend import facets


@pytest.mark.parametrize(
    "token,contrast,processing",
    [
        # April desc tags (task already peeled by parse_variant, tested below).
        ("bpfBOLD", "bold", "bpf"),
        ("bpfVASO", "vaso", "bpf"),
        ("optcom_bold", "bold", "optcom"),
        ("optcomMIRDenoised_bold", "bold", "optcomMIRdenoised"),
        ("MIRNoise_bold", "bold", "MIRnoise"),  # MIR noise is a BOLD-derived series
        # kw tokens.
        ("kwCBF4D", "cbf", "raw"),
        ("clean_kwCBF4D", "cbf", "clean"),
        ("kwCBF4D_MNI152_T1_2mm_brain", "cbf", "MNI152"),
        ("kwVASO4D", "vaso", "raw"),
        ("kwfcurN_Vaso", "vaso", "fcurN"),
        ("clean_kwfcurN_Vaso", "vaso", "clean"),
        ("kwBOLD4D", "bold", "raw"),
        ("kwfurN_Bold", "bold", "furN"),
        ("clean_kwfurN_Bold", "bold", "clean"),
        ("kwoptcomMIRDenoised_bold", "bold", "optcomMIRdenoised"),
    ],
)
def test_split_variant(token, contrast, processing):
    assert facets.split_variant(token) == (contrast, processing)


def test_parse_variant_peels_task():
    assert facets.parse_variant("co2_bpfBOLD") == ("co2", "bold", "bpf")
    assert facets.parse_variant("rest_MIRNoise_bold") == ("rest", "bold", "MIRnoise")
    # A "clean_" head is NOT a task; the token keeps its full name.
    assert facets.parse_variant("clean_kwCBF4D") == (None, "cbf", "clean")
    assert facets.parse_variant("kwfurN_Bold") == (None, "bold", "furN")


def test_detect_contrast():
    # Contrast is the acquisition modality. Both MIR-derived BOLD series
    # (denoised signal + removed noise) are BOLD — the processing distinguishes them.
    assert facets.detect_contrast("MIRNoise_bold") == "bold"
    assert facets.detect_contrast("optcomMIRDenoised_bold") == "bold"
    assert facets.detect_contrast("kwCBF4D") == "cbf"
    assert facets.detect_contrast("something_unknown") is None


# The real per-dataset variant token sets (from /api/catalog on the live bundles).
_NOV = [
    "clean_kwCBF4D", "clean_kwfcurN_Vaso", "clean_kwfurN_Bold", "kwBOLD4D", "kwCBF4D",
    "kwCBF4D_MNI152_T1_2mm_brain", "kwVASO4D", "kwfcurN_Vaso", "kwfurN_Bold",
    "kwoptcomMIRDenoised_bold",
]
_HOX = ["kwCBF4D", "kwfcurN_Vaso", "kwfurN_Bold", "kwoptcomMIRDenoised_bold"]
_APRIL = ["bpfBOLD", "bpfVASO", "MIRNoise_bold", "optcom_bold", "optcomMIRDenoised_bold"]


@pytest.mark.parametrize("tokens", [_NOV, _HOX, _APRIL])
def test_processing_unique_within_contrast(tokens):
    seen: dict[tuple[str | None, str | None], str] = {}
    for tok in tokens:
        contrast, processing = facets.split_variant(tok)
        key = (contrast, processing)
        assert key not in seen, (
            f"{tok} collides with {seen.get(key)} on {key}"
        )
        seen[key] = tok
