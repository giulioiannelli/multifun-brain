"""Tests for the multiscale-aware comparison metrics in
:mod:`multifunbrain.analysis.lrg.compare`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from multifunbrain.analysis.lrg.compare import (
    bakers_gamma,
    compare_hierarchies,
    cophenetic_correlation,
    null_shift_distribution,
    per_leaf_cophenetic_shift,
    reduced_mutual_information,
    specific_heat_comparison,
)
from multifunbrain.analysis.lrg.partitions import linkage_at_tau_min


def _block_graph(seed: int = 0, weak: float = 0.08) -> nx.Graph:
    r = np.random.default_rng(seed)
    n_per, k = 8, 3
    n = n_per * k
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i // n_per) == (j // n_per)
            A[i, j] = A[j, i] = abs(r.normal(0.7 if same else weak, 0.03))
    return nx.from_numpy_array(A)


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


class TestLinkageAtTauMin:
    def test_returns_linkage_leaves_and_tau(self):
        g = _block_graph()
        Z, leaf_atlas, tau = linkage_at_tau_min(g)
        n = g.number_of_nodes()
        assert Z.shape == (n - 1, 4)
        assert leaf_atlas.shape == (n,)
        # Leaf atlas ids are the graph's own node ids, in node order.
        assert leaf_atlas.tolist() == list(g.nodes())
        assert tau > 0

    def test_empty_graph_raises(self):
        g = nx.Graph()
        g.add_nodes_from(range(4))  # no edges -> degenerate Laplacian
        with pytest.raises(RuntimeError):
            linkage_at_tau_min(g)


class TestCompareHierarchies:
    def test_identical_hierarchies(self):
        g = _block_graph(seed=3)
        Z, leaf, _ = linkage_at_tau_min(g)
        cmp = compare_hierarchies(Z, leaf, Z, leaf)
        assert cmp.n_common == g.number_of_nodes()
        assert cmp.bakers_gamma == pytest.approx(1.0)
        assert cmp.cophenetic_r == pytest.approx(1.0)
        assert np.allclose(cmp.shift, 0.0)
        assert not cmp.has_null

    def test_different_hierarchies_finite(self):
        Za, la, _ = linkage_at_tau_min(_block_graph(seed=1))
        Zb, lb, _ = linkage_at_tau_min(_block_graph(seed=2, weak=0.25))
        cmp = compare_hierarchies(Za, la, Zb, lb)
        assert cmp.n_common == 24
        assert -1.0 <= cmp.bakers_gamma <= 1.0
        assert np.all(np.isfinite(cmp.shift))
        assert cmp.shift.min() >= 0.0 and cmp.shift.max() <= 1.0

    def test_intersection_on_disjoint_node_sets(self):
        # Two graphs whose node ids only partly overlap -> compare on the overlap.
        ga = _block_graph(seed=1)
        gb = nx.relabel_nodes(_block_graph(seed=2), {i: i + 6 for i in range(24)})
        Za, la, _ = linkage_at_tau_min(ga)
        Zb, lb, _ = linkage_at_tau_min(gb)
        cmp = compare_hierarchies(Za, la, Zb, lb)
        assert cmp.n_common == 18  # nodes 6..23 shared
        assert np.all(np.isfinite(cmp.shift))


class TestNullShiftDistribution:
    def test_returns_samples_per_atlas(self, tmp_path):
        ga, gb = _block_graph(seed=1), _block_graph(seed=2, weak=0.2)
        null = null_shift_distribution(
            ga, "A", gb, "B", n_surrogates=4, cache_dir=tmp_path,
        )
        assert isinstance(null, dict)
        assert len(null) > 0
        # Each ROI accumulates up to n_surrogates samples.
        assert all(1 <= len(v) <= 4 for v in null.values())
        assert all(np.isfinite(x) for v in null.values() for x in v)

    def test_calibrated_shift_uses_null(self, tmp_path):
        Za, la, _ = linkage_at_tau_min(_block_graph(seed=1))
        Zb, lb, _ = linkage_at_tau_min(_block_graph(seed=2, weak=0.2))
        null = null_shift_distribution(
            _block_graph(seed=1), "A", _block_graph(seed=2, weak=0.2), "B",
            n_surrogates=5, cache_dir=tmp_path,
        )
        cmp = compare_hierarchies(Za, la, Zb, lb, null_per_atlas=null)
        assert cmp.has_null
        assert cmp.pooled_null.size > 0
        # Calibrated = observed - null_median where a null exists.
        mask = np.isfinite(cmp.null_median)
        assert np.allclose(
            cmp.calibrated[mask], cmp.shift[mask] - cmp.null_median[mask]
        )


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
