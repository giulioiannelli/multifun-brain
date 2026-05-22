"""Tests for the multiscale-aware comparison metrics in
:mod:`multifunbrain.analysis.lrg.compare`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from multifunbrain.analysis.lrg.compare import (
    bakers_gamma,
    cophenetic_correlation,
    per_leaf_cophenetic_shift,
    reduced_mutual_information,
    specific_heat_comparison,
)


def _toy_linkage(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return linkage(pts, method="average")


class TestReducedMutualInformation:
    def test_identical_partitions_at_large_n(self):
        # At N=300 with 3 balanced clusters, the finite-N correction
        # beta = (r-1)(r-1) * log(N) / (2N) is small (≈0.038), so RMI ≈ H(A).
        labels = np.repeat([1, 2, 3], 100)
        rmi = reduced_mutual_information(labels, labels)
        h = -sum((100 / 300) * np.log(100 / 300) for _ in range(3))
        assert rmi == pytest.approx(h, rel=0.05)

    def test_identical_partitions_returns_mi_minus_correction(self):
        # For small N the asymptotic correction is substantial — verify
        # the explicit formula RMI = H(A) - (r-1)(r-1) log(N) / (2N).
        labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        rmi = reduced_mutual_information(labels, labels)
        h = -sum((3 / 9) * np.log(3 / 9) for _ in range(3))
        beta = (3 - 1) * (3 - 1) * np.log(9) / (2 * 9)
        assert rmi == pytest.approx(h - beta, abs=1e-9)

    def test_random_independent_partitions_near_zero(self):
        rng = np.random.default_rng(0)
        n = 200
        a = rng.integers(0, 5, size=n)
        b = rng.integers(0, 5, size=n)
        rmi = reduced_mutual_information(a, b)
        assert abs(rmi) < 0.2

    def test_different_cluster_counts_not_degenerate(self):
        rng = np.random.default_rng(1)
        n = 60
        a = rng.integers(0, 3, size=n)
        b = rng.integers(0, 12, size=n)
        rmi = reduced_mutual_information(a, b)
        assert np.isfinite(rmi)


class TestCopheneticCorrelation:
    def test_identical_linkage_returns_one(self):
        link = _toy_linkage(20)
        r = cophenetic_correlation(link, link)
        assert r == pytest.approx(1.0)

    def test_shuffled_linkage_decorrelates(self):
        link_a = _toy_linkage(20, seed=0)
        link_b = _toy_linkage(20, seed=99)
        r = cophenetic_correlation(link_a, link_b)
        assert r < 0.95


class TestBakersGamma:
    def test_identical_linkage_returns_one(self):
        link = _toy_linkage(20)
        g = bakers_gamma(link, link)
        assert g == pytest.approx(1.0)

    def test_random_pair_in_range(self):
        link_a = _toy_linkage(15, seed=0)
        link_b = _toy_linkage(15, seed=42)
        g = bakers_gamma(link_a, link_b)
        assert -1.0 <= g <= 1.0


class TestPerLeafCopheneticShift:
    def test_identical_linkage_returns_zeros(self):
        link = _toy_linkage(20)
        d = per_leaf_cophenetic_shift(link, link)
        assert d.shape == (20,)
        assert np.allclose(d, 0.0)

    def test_random_linkages_finite_and_in_unit_range(self):
        link_a = _toy_linkage(30, seed=0)
        link_b = _toy_linkage(30, seed=99)
        d = per_leaf_cophenetic_shift(link_a, link_b)
        assert d.shape == (30,)
        assert np.all(np.isfinite(d))
        assert d.min() >= 0.0
        assert d.max() <= 1.0

    def test_leaf_indices_restricts_output(self):
        link_a = _toy_linkage(25, seed=0)
        link_b = _toy_linkage(25, seed=7)
        keep = np.array([0, 2, 5, 9, 11, 17, 22])
        d_sub = per_leaf_cophenetic_shift(
            link_a, link_b, leaf_indices_a=keep, leaf_indices_b=keep,
        )
        d_full = per_leaf_cophenetic_shift(link_a, link_b)
        assert d_sub.shape == (len(keep),)
        # Restricting should generally differ from picking rows of the full
        # vector, because rank is recomputed on the sub-matrix only.
        assert not np.allclose(d_sub, d_full[keep])

    def test_misaligned_leaves_with_permutation(self):
        # If we shuffle B's leaves but pass an inverse permutation, the
        # result should match the unshuffled identical-linkage case (all 0).
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(20, 4))
        link_a = linkage(pts, method="average")
        perm = rng.permutation(20)
        link_b = linkage(pts[perm], method="average")
        d = per_leaf_cophenetic_shift(
            link_a, link_b,
            leaf_indices_a=np.arange(20),
            leaf_indices_b=np.argsort(perm),
        )
        assert np.allclose(d, 0.0, atol=1e-9)

    def test_unnormalized_scales_with_n_pairs(self):
        link_a = _toy_linkage(15, seed=0)
        link_b = _toy_linkage(15, seed=1)
        d_norm = per_leaf_cophenetic_shift(link_a, link_b, normalize=True)
        d_raw = per_leaf_cophenetic_shift(link_a, link_b, normalize=False)
        n_pairs = 15 * 14 // 2
        assert np.allclose(d_raw, d_norm * n_pairs)


class TestSpecificHeatComparison:
    def test_identical_curves(self):
        tau = np.logspace(-2, 5, 200)
        c = np.exp(-((np.log10(tau)) ** 2))
        out = specific_heat_comparison(c, c, tau)
        assert out["delta_tau_star"] == 0.0
        assert out["area_abs_delta"] == 0.0

    def test_shifted_peak_recovers_delta(self):
        tau = np.logspace(-2, 5, 400)
        c_a = np.exp(-((np.log10(tau) - 1.0) ** 2))
        c_b = np.exp(-((np.log10(tau) - 1.5) ** 2))
        out = specific_heat_comparison(c_a, c_b, tau)
        assert out["log_delta_tau_star"] == pytest.approx(-0.5, abs=0.1)
        assert out["area_abs_delta"] > 0.0
