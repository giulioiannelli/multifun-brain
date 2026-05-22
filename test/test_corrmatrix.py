"""Tests for multifunbrain.analysis.corrmatrix."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from multifunbrain.analysis.partition import (
    adjusted_rand_index,
    compare_partition_sets,
)
from multifunbrain.io import load_correlation_matrix
from multifunbrain.preprocessing import (
    detect_dead_regions,
    marchenko_pastur_denoise,
    prepare_correlation_matrix,
)


def _make_corr(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.corrcoef(rng.standard_normal((n, 50)))


class TestLoadCorrelationMatrix:
    def test_load_npy(self, tmp_path: Path) -> None:
        mat = _make_corr(10)
        np.save(tmp_path / "c.npy", mat)
        loaded = load_correlation_matrix(tmp_path / "c.npy")
        np.testing.assert_array_almost_equal(loaded, mat)

    def test_load_csv(self, tmp_path: Path) -> None:
        mat = _make_corr(8)
        np.savetxt(tmp_path / "c.csv", mat, delimiter=",")
        loaded = load_correlation_matrix(tmp_path / "c.csv")
        np.testing.assert_array_almost_equal(loaded, mat, decimal=6)

    def test_load_pkl(self, tmp_path: Path) -> None:
        mat = _make_corr(6)
        with open(tmp_path / "c.pkl", "wb") as f:
            pickle.dump(mat, f)
        loaded = load_correlation_matrix(tmp_path / "c.pkl")
        np.testing.assert_array_almost_equal(loaded, mat)

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_correlation_matrix(tmp_path / "no.npy")

    def test_load_unsupported(self, tmp_path: Path) -> None:
        (tmp_path / "bad.xyz").write_text("x")
        with pytest.raises(ValueError):
            load_correlation_matrix(tmp_path / "bad.xyz")


class TestPrepareCorrelationMatrix:
    def test_symmetrize(self) -> None:
        mat = np.array([[1.0, 0.8, 0.2], [0.3, 1.0, 0.5], [0.1, 0.9, 1.0]])
        result = prepare_correlation_matrix(mat)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_clip(self) -> None:
        mat = np.array([[1.0, 1.5, -1.3], [1.5, 1.0, 0.4], [-1.3, 0.4, 1.0]])
        result = prepare_correlation_matrix(mat, clip=True)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_diagonal(self) -> None:
        result = prepare_correlation_matrix(_make_corr(10), zero_diagonal=True)
        np.testing.assert_array_equal(np.diag(result), 0.0)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            prepare_correlation_matrix(np.ones((3, 5)))


class TestDenoise:
    def test_shape_preserved(self) -> None:
        mat = _make_corr(12)
        assert marchenko_pastur_denoise(mat, gamma=0.5).shape == mat.shape


class TestARI:
    def test_identical(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            adjusted_rand_index([0, 1, 2], [0, 1])


class TestComparePartitions:
    def test_length(self) -> None:
        a = [{"tau": 0.1, "partition": np.array([0, 0, 1, 1])}]
        b = [
            {"tau": 0.5, "partition": np.array([0, 0, 1, 1])},
            {"tau": 1.0, "partition": np.array([1, 1, 0, 0])},
        ]
        assert len(compare_partition_sets(a, b)) == 2


def _make_nan_matrix(n: int = 10, dead_indices: list | None = None) -> np.ndarray:
    """Create a correlation matrix with dead regions (entire NaN rows/cols)."""
    rng = np.random.default_rng(42)
    mat = np.corrcoef(rng.standard_normal((n, 50)))
    if dead_indices is None:
        dead_indices = []
    for i in dead_indices:
        mat[i, :] = np.nan
        mat[:, i] = np.nan
    return mat


class TestDetectDeadRegions:
    def test_no_dead(self) -> None:
        mat = _make_corr(10)
        dead = detect_dead_regions(mat)
        assert len(dead) == 0

    def test_one_dead(self) -> None:
        mat = _make_nan_matrix(10, dead_indices=[3])
        dead = detect_dead_regions(mat)
        np.testing.assert_array_equal(dead, [3])

    def test_multiple_dead(self) -> None:
        mat = _make_nan_matrix(10, dead_indices=[1, 5, 8])
        dead = detect_dead_regions(mat)
        np.testing.assert_array_equal(dead, [1, 5, 8])

    def test_empty_matrix(self) -> None:
        mat = np.zeros((0, 0))
        dead = detect_dead_regions(mat)
        assert len(dead) == 0

    def test_non_square_returns_empty(self) -> None:
        dead = detect_dead_regions(np.ones((3, 5)))
        assert len(dead) == 0

    def test_partial_nan_not_dead(self) -> None:
        """A region with only a few NaN entries is NOT dead at default threshold."""
        mat = _make_corr(10)
        mat[2, 3] = np.nan
        mat[3, 2] = np.nan
        dead = detect_dead_regions(mat)
        assert len(dead) == 0


class TestPrepareDropsDeadRegions:
    def test_drops_single_dead_region(self) -> None:
        mat = _make_nan_matrix(10, dead_indices=[7])
        result = prepare_correlation_matrix(mat)
        # Should be 9x9 (dropped 1 region)
        assert result.shape == (9, 9)
        assert np.all(np.isfinite(result))

    def test_drops_multiple_dead_regions(self) -> None:
        mat = _make_nan_matrix(10, dead_indices=[2, 6])
        result = prepare_correlation_matrix(mat)
        assert result.shape == (8, 8)
        assert np.all(np.isfinite(result))

    def test_clean_matrix_unchanged(self) -> None:
        mat = _make_corr(10)
        result = prepare_correlation_matrix(mat)
        # Shape should be preserved (no dead regions)
        assert result.shape == (10, 10)

    def test_all_dead_regions(self) -> None:
        mat = np.full((5, 5), np.nan)
        result = prepare_correlation_matrix(mat)
        assert result.shape == (0, 0)
