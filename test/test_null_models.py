"""Tests for the strength-preserving null model in
:mod:`multifunbrain.processing.null_models`.

The headline property under test is that
:func:`strength_preserving_rewire` preserves each node's binary degree
**exactly** and its weighted strength **approximately** (mean relative
error < 5 %) on graphs of LANS-backbone scale (~100 nodes, ~500
edges).
"""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest

from multifunbrain.processing.null_models import (
    cached_surrogate_linkages,
    strength_preserving_rewire,
)


def _weighted_random_graph(n: int = 60, p: float = 0.18, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.05, 1.0))
    return G


def _strength(G: nx.Graph) -> np.ndarray:
    nodes = sorted(G.nodes())
    return np.array(
        [sum(d.get("weight", 0.0) for _, d in G[nd].items()) for nd in nodes],
        dtype=float,
    )


def _degree(G: nx.Graph) -> np.ndarray:
    nodes = sorted(G.nodes())
    return np.array([G.degree(nd) for nd in nodes], dtype=int)


class TestStrengthPreservingRewire:
    def test_zero_swaps_keeps_edge_set(self):
        G = _weighted_random_graph(n=40, p=0.2, seed=1)
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng, n_swaps=0, weight_iters=0)
        # Topology unchanged, weights are a permutation of the originals.
        assert set(G_out.edges()) == set(G.edges())
        w_in = sorted(d["weight"] for _, _, d in G.edges(data=True))
        w_out = sorted(d["weight"] for _, _, d in G_out.edges(data=True))
        np.testing.assert_allclose(w_in, w_out)

    def test_preserves_degree_exactly(self):
        G = _weighted_random_graph(n=80, p=0.15, seed=2)
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng)
        np.testing.assert_array_equal(_degree(G_out), _degree(G))

    def test_preserves_strength_within_tolerance(self):
        G = _weighted_random_graph(n=80, p=0.15, seed=3)
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng)
        s_in = _strength(G)
        s_out = _strength(G_out)
        rel_err = np.abs(s_out - s_in) / np.maximum(s_in, 1e-12)
        assert rel_err.mean() < 0.05
        assert rel_err.max() < 0.20

    def test_destroys_topology(self):
        G = _weighted_random_graph(n=80, p=0.15, seed=4)
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng)
        common = set(G.edges()) & set(G_out.edges())
        assert len(common) / G.number_of_edges() < 0.7

    def test_deterministic_with_seed(self):
        G = _weighted_random_graph(n=50, p=0.18, seed=5)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        G_a = strength_preserving_rewire(G, rng_a)
        G_b = strength_preserving_rewire(G, rng_b)
        assert set(G_a.edges()) == set(G_b.edges())
        for u, v in G_a.edges():
            assert G_a[u][v]["weight"] == pytest.approx(G_b[u][v]["weight"])

    def test_total_weight_preserved(self):
        G = _weighted_random_graph(n=60, p=0.18, seed=6)
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng)
        w_in = sum(d["weight"] for _, _, d in G.edges(data=True))
        w_out = sum(d["weight"] for _, _, d in G_out.edges(data=True))
        assert w_out == pytest.approx(w_in)

    def test_empty_graph_returns_empty(self):
        G = nx.Graph()
        G.add_nodes_from(range(10))
        rng = np.random.default_rng(0)
        G_out = strength_preserving_rewire(G, rng)
        assert G_out.number_of_edges() == 0
        assert set(G_out.nodes()) == set(G.nodes())


class TestCachedSurrogateLinkages:
    def test_cache_resumes_extension(self, tmp_path):
        G = _weighted_random_graph(n=40, p=0.2, seed=7)

        def fake_linkage(graph):
            n = graph.number_of_nodes()
            Z = np.column_stack(
                [
                    np.arange(n - 1, dtype=float),
                    np.arange(1, n, dtype=float),
                    np.linspace(0.1, 1.0, n - 1),
                    np.full(n - 1, 2.0),
                ]
            )
            nodes = np.array(sorted(graph.nodes()), dtype=int)
            return Z, nodes, 0.5

        first = cached_surrogate_linkages(
            G, "test", n_surrogates=5,
            cache_dir=tmp_path, linkage_fn=fake_linkage, rng_seed=0,
        )
        assert len(first) == 5
        cache_path = tmp_path / "test__surrogates.pkl"
        assert cache_path.exists()
        with open(cache_path, "rb") as f:
            persisted = pickle.load(f)
        assert len(persisted) == 5

        extended = cached_surrogate_linkages(
            G, "test", n_surrogates=8,
            cache_dir=tmp_path, linkage_fn=fake_linkage, rng_seed=0,
        )
        assert len(extended) == 8
        for a, b in zip(first, extended[:5]):
            np.testing.assert_array_equal(a["linkage"], b["linkage"])

    def test_cache_truncates_on_request(self, tmp_path):
        G = _weighted_random_graph(n=40, p=0.2, seed=8)

        def fake_linkage(graph):
            n = graph.number_of_nodes()
            Z = np.column_stack(
                [
                    np.arange(n - 1, dtype=float),
                    np.arange(1, n, dtype=float),
                    np.linspace(0.1, 1.0, n - 1),
                    np.full(n - 1, 2.0),
                ]
            )
            nodes = np.array(sorted(graph.nodes()), dtype=int)
            return Z, nodes, 0.5

        big = cached_surrogate_linkages(
            G, "test", n_surrogates=6,
            cache_dir=tmp_path, linkage_fn=fake_linkage, rng_seed=0,
        )
        small = cached_surrogate_linkages(
            G, "test", n_surrogates=3,
            cache_dir=tmp_path, linkage_fn=fake_linkage, rng_seed=0,
        )
        assert len(small) == 3
        for a, b in zip(big[:3], small):
            np.testing.assert_array_equal(a["linkage"], b["linkage"])
