"""Aggregate network-analysis report orchestrator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .community import detect_communities_louvain
from .distribution import compute_rich_club_curve, degree_distribution_analysis
from .global_metrics import compute_global_metrics
from .node_metrics import compute_node_metrics

__all__ = ["network_summary_report"]


def network_summary_report(
    G: nx.Graph,
    weight: str | None = "weight",
    run_community_detection: bool = True,
    run_rich_club: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Produce a comprehensive summary report for a network.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str or None
        Edge weight attribute.
    run_community_detection : bool
        If True, runs Louvain and includes partition + modularity.
    run_rich_club : bool
        If True, computes the rich-club curve (can be slow).
    seed : int or None
        Random seed for stochastic algorithms.

    Returns
    -------
    dict
        Nested dict with ``'global_metrics'``, ``'node_metrics'``,
        ``'community'``, ``'degree_distribution'``, ``'rich_club'``.
    """
    report: dict[str, Any] = {
        "global_metrics": compute_global_metrics(G, weight=weight),
        "degree_distribution": degree_distribution_analysis(G, weight=weight),
    }

    partition = None
    if run_community_detection and G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        partition = detect_communities_louvain(G, weight=weight, seed=seed)
        modularity = nx.community.modularity(
            G,
            [
                {n for n, c in partition.items() if c == cid}
                for cid in set(partition.values())
            ],
            weight=weight,
        )
        report["community"] = {
            "partition": partition,
            "n_communities": len(set(partition.values())),
            "modularity": modularity,
        }
    elif run_community_detection and G.number_of_nodes() > 0:
        report["community"] = {
            "partition": {n: 0 for n in G.nodes()},
            "n_communities": 0,
            "modularity": float("nan"),
        }

    report["node_metrics"] = compute_node_metrics(
        G, weight=weight, community_partition=partition
    )

    if run_rich_club:
        report["rich_club"] = compute_rich_club_curve(G, seed=seed)

    return report
