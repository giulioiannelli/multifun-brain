"""Tests for the LRG dashboard serializers (``dashboard.backend.serializers.lrg``).

Covers the specific-heat ``C(τ)`` curve (recomputed online from the filtered
graph's Laplacian spectrum), the partition-stability ``Ψ(n)`` curve, and the
dendrogram payload's client-side-colouring contract (linkage + per-link node ids,
no server-side ``color_list`` / ``cut_height``). A real ``PipelineResult`` is built
from a synthetic 3-block correlation matrix so the test needs no gitignored data.
"""

from __future__ import annotations

import numpy as np
import pytest

from dashboard.backend.serializers import lrg
from multifunbrain import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def result():
    # Three strongly-correlated blocks -> a non-trivial diffusion hierarchy.
    rng = np.random.default_rng(0)
    n_per, k, n_t = 12, 3, 300
    latent = rng.normal(size=(k, n_t))
    x = np.empty((n_per * k, n_t))
    for b in range(k):
        x[b * n_per : (b + 1) * n_per] = latent[b] + 0.3 * rng.normal(size=(n_per, n_t))
    corr = np.corrcoef(x)
    return run_pipeline(corr, PipelineConfig(), label="global/test_bold")


def _filter(result) -> str:
    return next(iter(result.lrg_results))


def test_specific_heat_spec(result):
    spec = lrg.specific_heat_spec(result, filter_name=_filter(result))
    assert not spec.get("error"), spec
    assert spec["kind"] == "specific_heat"
    assert len(spec["C"]) == len(spec["tau"]) > 100
    assert all(t > 0 for t in spec["tau"])
    # C(τ) has a genuine peak (structure), and τ* / τ′ are marked.
    assert max(spec["C"]) > 0
    assert spec["tau_star"] and spec["tau_star"] > 0
    assert spec["tau_prime"] and spec["tau_prime"] > 0


def test_psi_spec(result):
    spec = lrg.psi_spec(result, filter_name=_filter(result), tau_index=-1)
    assert not spec.get("error"), spec
    assert spec["kind"] == "psi"
    assert len(spec["psi"]) == len(spec["n_clusters"]) == len(spec["heights"]) >= 1
    assert spec["n_clusters"][0] == 2  # branch 0 -> 2 clusters
    assert spec["optimal_n_clusters"] >= 2
    # optimal_psi is the peak of the Ψ curve.
    assert spec["optimal_psi"] == pytest.approx(max(spec["psi"]))


def test_dendrogram_client_side_colouring_contract(result):
    spec = lrg.dendrogram_spec(result, filter_name=_filter(result), tau_index=-1)
    assert not spec.get("error"), spec
    # The server no longer colours: coordinates + linkage + per-link node ids only.
    assert "color_list" not in spec
    assert "cut_height" not in spec
    n_leaves = spec["n_leaves"]
    assert len(spec["linkage"]) == n_leaves - 1
    assert len(spec["leaves"]) == n_leaves
    assert len(spec["link_ids"]) == len(spec["icoord"]) == len(spec["dcoord"])
    # link ids are internal (non-singleton) linkage node ids: n_leaves .. 2n-2.
    assert all(n_leaves <= lid <= 2 * n_leaves - 2 for lid in spec["link_ids"])
    assert spec["flat_threshold"] is not None
    assert spec["dcoord_max"] >= spec["dcoord_min_positive"] > 0


def test_specific_heat_graceful_without_graph(result):
    # A result whose filter has no stored graph -> graceful error, no crash.
    import copy

    r = copy.copy(result)
    r.filtered_networks = {}
    spec = lrg.specific_heat_spec(r, filter_name=_filter(result))
    assert spec["error"]
    assert spec["kind"] == "specific_heat"
