"""Tests for multifunbrain.analysis.netmetrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from multifunbrain.analysis.netmetrics import (
    compute_global_metrics,
    compute_node_metrics,
    compute_rich_club_curve,
    degree_distribution_analysis,
    detect_communities_louvain,
    detect_communities_spectral,
    network_summary_report,
)


def _complete(n: int = 5) -> nx.Graph:
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def _disconnected() -> nx.Graph:
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    return G


class TestGlobalMetrics:
    def test_complete(self) -> None:
        gm = compute_global_metrics(_complete(5))
        assert gm["density"] == pytest.approx(1.0)
        assert gm["is_connected"] is True

    def test_disconnected(self) -> None:
        gm = compute_global_metrics(_disconnected())
        assert np.isnan(gm["avg_shortest_path"])
        assert gm["n_components"] == 2

    def test_empty(self) -> None:
        gm = compute_global_metrics(nx.Graph())
        assert gm["n_nodes"] == 0


class TestNodeMetrics:
    def test_columns(self) -> None:
        df = compute_node_metrics(_complete())
        for col in ("degree", "strength", "clustering", "betweenness", "closeness"):
            assert col in df.columns

    def test_with_partition(self) -> None:
        G = _complete()
        partition = {n: n % 2 for n in G.nodes()}
        df = compute_node_metrics(G, community_partition=partition)
        assert "participation_coefficient" in df.columns
        assert "within_module_z" in df.columns


class TestCommunityDetection:
    def test_louvain_all_nodes(self) -> None:
        G = _complete(10)
        part = detect_communities_louvain(G, seed=42)
        assert set(part.keys()) == set(G.nodes())

    def test_spectral(self) -> None:
        G = _complete(10)
        part = detect_communities_spectral(G)
        assert set(part.keys()) == set(G.nodes())


class TestDegreeDistribution:
    def test_keys(self) -> None:
        dd = degree_distribution_analysis(_complete())
        for key in ("values", "mean", "std", "histogram"):
            assert key in dd


class TestRichClub:
    def test_keys(self) -> None:
        rc = compute_rich_club_curve(_complete(10), normalized=False)
        assert "k" in rc
        assert "phi" in rc


class TestSummaryReport:
    def test_keys(self) -> None:
        report = network_summary_report(_complete(), seed=42)
        assert "global_metrics" in report
        assert "node_metrics" in report
        assert "community" in report
        assert isinstance(report["node_metrics"], pd.DataFrame)
