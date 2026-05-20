"""Tests for multifunbrain.analysis.filtering."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from multifunbrain.analysis.filtering import (
    apply_all_filters,
    filter_absolute_threshold,
    filter_partial_correlation,
    filter_split_sign,
    filter_validated,
)


def _make_corr(n: int = 15) -> np.ndarray:
    rng = np.random.default_rng(42)
    C = np.corrcoef(rng.standard_normal((n, 80)))
    np.fill_diagonal(C, 0.0)
    return C


class TestAbsoluteThreshold:
    def test_basic(self) -> None:
        G, rem = filter_absolute_threshold(_make_corr(), threshold=0.0)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0

    def test_higher_threshold_fewer_edges(self) -> None:
        G_low, _ = filter_absolute_threshold(_make_corr(), threshold=0.0)
        G_high, _ = filter_absolute_threshold(_make_corr(), threshold=0.3)
        assert G_high.number_of_edges() <= G_low.number_of_edges()


class TestSplitSign:
    def test_both_returned(self) -> None:
        result = filter_split_sign(_make_corr())
        assert "positive" in result
        assert "negative" in result
        assert isinstance(result["positive"], nx.Graph)

    def test_positive_weights_positive(self) -> None:
        result = filter_split_sign(_make_corr())
        for _, _, w in result["positive"].edges(data="weight"):
            assert w > 0


class TestValidated:
    def test_disparity(self) -> None:
        G, _ = filter_validated(_make_corr(), method="disparity", alpha=0.1)
        assert isinstance(G, nx.Graph)

    def test_mp_validated_requires_gamma(self) -> None:
        with pytest.raises(ValueError):
            filter_validated(_make_corr(), method="mp_validated")


class TestPartialCorrelation:
    def test_basic(self) -> None:
        C = _make_corr(10)
        np.fill_diagonal(C, 1.0)
        G, _ = filter_partial_correlation(C)
        assert isinstance(G, nx.Graph)


class TestApplyAllFilters:
    def test_keys(self) -> None:
        result = apply_all_filters(
            _make_corr(),
            methods=["absolute", "positive"],
        )
        assert "absolute" in result
        assert "positive" in result
        assert "graph" in result["absolute"]
