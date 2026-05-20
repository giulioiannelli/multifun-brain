"""Tests for multifunbrain.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multifunbrain.pipeline import PipelineConfig, PipelineResult, run_pipeline, run_pipeline_batch


def _make_corr(n: int = 15) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.corrcoef(rng.standard_normal((n, 80)))


class TestRunPipeline:
    def test_from_array(self) -> None:
        result = run_pipeline(
            _make_corr(),
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
            label="test",
        )
        assert isinstance(result, PipelineResult)
        assert result.label == "test"
        assert "absolute" in result.filtered_networks
        assert "absolute" in result.network_analyses

    def test_with_gamma(self) -> None:
        result = run_pipeline(
            _make_corr(),
            config=PipelineConfig(
                gamma=0.19,
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
        )
        assert result.descriptive["spectrum"]["n_signal"] >= 0

    def test_no_lrg(self) -> None:
        result = run_pipeline(
            _make_corr(),
            config=PipelineConfig(
                filter_methods=["positive"],
                run_lrg=False,
                seed=42,
            ),
        )
        assert len(result.lrg_results) == 0


class TestRunPipelineBatch:
    def test_batch(self) -> None:
        mats = [_make_corr(10), _make_corr(10), _make_corr(10)]
        results = run_pipeline_batch(
            mats,
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
            labels=["a", "b", "c"],
        )
        assert len(results) == 3
        assert results[0].label == "a"


class TestPipelineWithDeadRegions:
    def test_dead_regions_tracked(self) -> None:
        """Pipeline correctly detects and drops dead brain regions."""
        mat = _make_corr(15)
        # Simulate dead region at index 3
        mat[3, :] = np.nan
        mat[:, 3] = np.nan
        result = run_pipeline(
            mat,
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
            label="nan_test",
        )
        assert result.n_regions_original == 15
        assert len(result.dropped_regions) == 1
        assert 3 in result.dropped_regions
        # Prepared matrix should be 14x14
        assert result.corr_prepared.shape == (14, 14)
        # Pipeline should still complete successfully
        assert "absolute" in result.network_analyses

    def test_no_dead_regions(self) -> None:
        result = run_pipeline(
            _make_corr(10),
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
        )
        assert result.n_regions_original == 10
        assert len(result.dropped_regions) == 0

    def test_to_dict_includes_dropped(self) -> None:
        mat = _make_corr(10)
        mat[2, :] = np.nan
        mat[:, 2] = np.nan
        result = run_pipeline(
            mat,
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
        )
        d = result.to_dict()
        assert d["n_regions_original"] == 10
        assert d["n_regions_after_cleanup"] == 9
        assert 2 in d["dropped_regions"]


class TestPipelineResult:
    def test_to_dict(self) -> None:
        result = run_pipeline(
            _make_corr(10),
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
            label="dict_test",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["label"] == "dict_test"

    def test_summary_table(self) -> None:
        result = run_pipeline(
            _make_corr(10),
            config=PipelineConfig(
                filter_methods=["absolute"],
                run_lrg=False,
                seed=42,
            ),
        )
        df = result.summary_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "density" in df.columns
