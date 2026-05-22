"""Tests for multifunbrain.analysis.descriptive."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.analysis.descriptive import (
    compute_precision_matrix,
    correlation_spectrum_analysis,
    descriptive_report,
    signed_laplacian_analysis,
    signed_network_metrics,
    weight_distribution_analysis,
)


def _make_corr(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(42)
    C = np.corrcoef(rng.standard_normal((n, 50)))
    np.fill_diagonal(C, 0.0)
    return C


class TestWeightDistribution:
    def test_keys(self) -> None:
        wd = weight_distribution_analysis(_make_corr(8))
        for key in ("n_positive", "n_negative", "mean", "std", "skewness", "kurtosis"):
            assert key in wd

    def test_counts_consistent(self) -> None:
        wd = weight_distribution_analysis(_make_corr(10))
        assert wd["n_positive"] + wd["n_negative"] + wd["n_zero"] == wd["n_total"]


class TestSpectrumAnalysis:
    def test_no_gamma(self) -> None:
        spec = correlation_spectrum_analysis(_make_corr(8))
        assert "eigenvalues" in spec
        assert "mp_lambda_plus" not in spec

    def test_with_gamma(self) -> None:
        spec = correlation_spectrum_analysis(_make_corr(10), gamma=0.2)
        assert "mp_lambda_plus" in spec
        assert spec["n_signal"] + spec["n_noise"] <= 10


class TestPrecisionMatrix:
    def test_direct_shape(self) -> None:
        C = _make_corr(8)
        np.fill_diagonal(C, 1.0)
        result = compute_precision_matrix(C, method="direct")
        assert result["precision_matrix"].shape == C.shape

    def test_orie_requires_gamma(self) -> None:
        C = _make_corr(8)
        np.fill_diagonal(C, 1.0)
        with pytest.raises(ValueError):
            compute_precision_matrix(C, method="orie")

    def test_orie_works(self) -> None:
        C = _make_corr(8)
        np.fill_diagonal(C, 1.0)
        result = compute_precision_matrix(C, method="orie", gamma=0.16)
        assert result["precision_matrix"].shape == C.shape


class TestSignedLaplacian:
    def test_keys(self) -> None:
        sl = signed_laplacian_analysis(_make_corr(10))
        for key in ("L_signed", "eigenvalues", "n_negative_eigenvalues", "frustration_index"):
            assert key in sl

    def test_spectrum_length(self) -> None:
        sl = signed_laplacian_analysis(_make_corr(10))
        assert len(sl["eigenvalues"]) == 10


class TestSignedNetworkMetrics:
    def test_keys(self) -> None:
        nm = signed_network_metrics(_make_corr(10))
        for key in ("n_nodes", "n_edges", "density", "balance_ratio", "mean_strength"):
            assert key in nm

    def test_n_nodes(self) -> None:
        assert signed_network_metrics(_make_corr(12))["n_nodes"] == 12


class TestDescriptiveReport:
    def test_sections(self) -> None:
        report = descriptive_report(_make_corr(8))
        for key in ("weight_distribution", "spectrum", "precision", "signed_laplacian", "network_metrics"):
            assert key in report
