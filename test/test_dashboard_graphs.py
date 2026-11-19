"""Tests for the shared graph derivation + LRG-cut colouring (dashboard backend).

Covers ``dashboard.backend.graphs.derive_graph`` (the single sparsify dispatcher
used by Network / LRG / Brain-3D) including the LANS and MP-validated backbones,
and ``glassbrain.leaf_cluster_ids`` (the server mirror of the frontend dendrogram
colouring). A real ``PipelineResult`` is built from synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

from dashboard.backend.glassbrain import leaf_cluster_ids
from dashboard.backend.graphs import derive_graph, result_gamma
from multifunbrain import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    n_per, k, n_t = 12, 3, 200
    latent = rng.normal(size=(k, n_t))
    x = np.empty((n_per * k, n_t))
    for b in range(k):
        x[b * n_per : (b + 1) * n_per] = latent[b] + 0.3 * rng.normal(size=(n_per, n_t))
    corr = np.corrcoef(x)
    # gamma set so mp_validated is available.
    return run_pipeline(corr, PipelineConfig(gamma=(n_per * k) / n_t), label="global/test_bold")


def _fname(result) -> str:
    return next(iter(result.filtered_networks))


class TestDeriveGraph:
    def test_filter_returns_stored(self, result):
        fname = _fname(result)
        g, eff = derive_graph(result, fname, "filter", 0.05, 0.3)
        assert eff == "filter"
        assert g is result.filtered_networks[fname]["graph"]

    def test_lans_backbone(self, result):
        fname = _fname(result)
        g, eff = derive_graph(result, fname, "lans", 0.05, 0.3)
        assert eff == "lans"
        assert g.number_of_nodes() > 0 and g.number_of_edges() > 0
        # LANS on positive weights -> non-negative edges.
        assert all(d["weight"] >= 0 for _, _, d in g.edges(data=True))

    def test_mp_validated_needs_gamma(self, result):
        fname = _fname(result)
        assert result_gamma(result) is not None
        g, eff = derive_graph(result, fname, "mp_validated", 0.05, 0.3)
        # Either a real backbone, or a graceful fall back to the stored network.
        assert eff in ("mp_validated", "filter")
        assert g.number_of_nodes() > 0

    def test_unknown_falls_back(self, result):
        fname = _fname(result)
        g, eff = derive_graph(result, fname, "nonsense", 0.05, 0.3)
        assert eff == "filter"
        assert g is result.filtered_networks[fname]["graph"]


class TestLeafClusterIds:
    def test_matches_fcluster_partition(self):
        rng = np.random.default_rng(0)
        Z = linkage(rng.normal(size=(20, 4)), method="average")
        h = float(np.median(Z[:, 2]))
        ids, k = leaf_cluster_ids(Z, h)
        fc = fcluster(Z, h, criterion="distance")

        def groups(a):
            from collections import defaultdict

            d = defaultdict(set)
            for i, c in enumerate(a):
                d[c].add(i)
            return sorted(map(frozenset, d.values()), key=lambda s: min(s))

        assert groups(ids) == groups(fc)
        assert k == len(set(fc))

    def test_ids_ordered_by_leftmost_leaf(self):
        # The cluster containing the FIRST leaf in dendrogram order gets colour 0.
        from scipy.cluster.hierarchy import dendrogram

        rng = np.random.default_rng(1)
        Z = linkage(rng.normal(size=(15, 3)), method="average")
        h = float(np.percentile(Z[:, 2], 40))
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        ids, _ = leaf_cluster_ids(Z, h)
        assert ids[leaves[0]] == 0

    def test_full_merge_single_cluster(self):
        rng = np.random.default_rng(2)
        Z = linkage(rng.normal(size=(10, 3)), method="average")
        ids, k = leaf_cluster_ids(Z, Z[:, 2].max() + 1.0)
        assert k == 1
        assert set(ids.tolist()) == {0}
