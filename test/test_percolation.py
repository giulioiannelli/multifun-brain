"""Tests for the percolation helpers in multifunbrain.processing.percolation."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.processing.percolation import (
    percolation_curve,
    percolation_threshold,
)


def _block_network(n_per_block: int = 6, n_blocks: int = 3, inter: float = 0.05) -> np.ndarray:
    """Build a modular non-negative weight matrix: strong intra-block, weak inter-block."""
    n = n_per_block * n_blocks
    rng = np.random.default_rng(0)
    A = rng.uniform(0.0, inter, size=(n, n))
    for b in range(n_blocks):
        i0 = b * n_per_block
        i1 = i0 + n_per_block
        A[i0:i1, i0:i1] = rng.uniform(0.7, 1.0, size=(n_per_block, n_per_block))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


class TestPercolationThreshold:
    def test_returns_min_of_max_per_node(self) -> None:
        A = _block_network()
        th, info = percolation_threshold(A)
        assert 0.0 <= th <= float(A.max())
        assert 0 <= info["first_detached"] < A.shape[0]
        assert 0.0 <= info["p_inf"] <= 1.0


class TestPercolationCurve:
    def test_shape_and_endpoints(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=20)
        assert out["thresholds"].shape == (20,)
        assert out["p_inf"].shape == (20,)
        assert out["e_inf"].shape == (20,)
        assert out["thresholds"][0] == 0.0
        assert out["thresholds"][-1] == pytest.approx(float(A.max()))

    def test_p_inf_monotone_non_increasing(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=30)
        diffs = np.diff(out["p_inf"])
        assert np.all(diffs <= 1e-12), (
            "P_inf must be non-increasing as the threshold rises"
        )

    def test_p_inf_starts_full(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=15)
        assert out["p_inf"][0] == pytest.approx(1.0)

    def test_p_inf_collapses_at_max_threshold(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=15)
        assert out["p_inf"][-1] <= 2.0 / A.shape[0] + 1e-12

    def test_skip_e_inf(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=10, compute_e_inf=False)
        assert np.all(np.isnan(out["e_inf"]))

    def test_empty_matrix(self) -> None:
        out = percolation_curve(np.zeros((0, 0)), n_thresholds=5)
        assert out["p_inf"].shape == (5,)
        assert np.all(out["p_inf"] == 0.0)

    def test_zero_weight_matrix(self) -> None:
        A = np.zeros((6, 6))
        out = percolation_curve(A, n_thresholds=5)
        assert np.all(out["p_inf"] == 0.0)
        assert np.all(out["e_inf"] == 0.0) or np.all(np.isnan(out["e_inf"]))

    def test_custom_threshold_range(self) -> None:
        A = _block_network()
        out = percolation_curve(A, n_thresholds=10, threshold_range=(0.2, 0.6))
        assert out["thresholds"][0] == pytest.approx(0.2)
        assert out["thresholds"][-1] == pytest.approx(0.6)
