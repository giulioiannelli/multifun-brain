"""Per-node centrality and connectivity metrics for unsigned graphs.

Also implements participation coefficient and within-module degree
z-score (Guimera & Amaral, 2005) when a community partition is provided.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

__all__ = ["compute_node_metrics"]


def compute_node_metrics(
    G: nx.Graph,
    weight: str | None = "weight",
    community_partition: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Compute per-node centrality and connectivity metrics.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str or None
        Edge attribute for weighted computations.
    community_partition : dict or None
        Mapping ``{node: community_id}``. If provided, also computes
        participation coefficient and within-module degree z-score.

    Returns
    -------
    pd.DataFrame
        Indexed by node.
    """
    nodes = list(G.nodes())
    data: dict[str, dict] = {}

    data["degree"] = dict(G.degree())
    data["strength"] = dict(G.degree(weight=weight))
    data["clustering"] = nx.clustering(G, weight=weight)
    data["betweenness"] = nx.betweenness_centrality(G, weight=weight)
    data["closeness"] = nx.closeness_centrality(G, distance=weight)

    try:
        data["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(
            G, weight=weight
        )
    except (nx.NetworkXError, nx.AmbiguousSolution, np.linalg.LinAlgError, TypeError):
        data["eigenvector_centrality"] = {n: float("nan") for n in nodes}

    df = pd.DataFrame(data, index=nodes)

    if community_partition is not None:
        df["participation_coefficient"] = pd.Series(
            _participation_coefficient(G, community_partition, weight)
        )
        df["within_module_z"] = pd.Series(
            _within_module_degree_z(G, community_partition, weight)
        )

    return df


def _participation_coefficient(
    G: nx.Graph,
    partition: dict[int, int],
    weight: str | None = "weight",
) -> dict[int, float]:
    """Participation coefficient (Guimera & Amaral 2005)."""
    result = {}
    for node in G.nodes():
        ki = G.degree(node, weight=weight)
        if ki == 0:
            result[node] = 0.0
            continue
        community_strength: dict[int, float] = {}
        for neighbour in G.neighbors(node):
            c = partition.get(neighbour)
            w = G[node][neighbour].get(weight, 1.0) if weight else 1.0
            community_strength[c] = community_strength.get(c, 0.0) + w
        result[node] = 1.0 - sum((ks / ki) ** 2 for ks in community_strength.values())
    return result


def _within_module_degree_z(
    G: nx.Graph,
    partition: dict[int, int],
    weight: str | None = "weight",
) -> dict[int, float]:
    """Within-module degree z-score (Guimera & Amaral 2005)."""
    communities: dict[int, list[int]] = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    within_degree: dict[int, float] = {}
    for node in G.nodes():
        comm = partition.get(node)
        wd = 0.0
        for neighbour in G.neighbors(node):
            if partition.get(neighbour) == comm:
                wd += G[node][neighbour].get(weight, 1.0) if weight else 1.0
        within_degree[node] = wd

    result: dict[int, float] = {}
    for _comm, members in communities.items():
        wds = np.array([within_degree[m] for m in members])
        mean_wd = np.mean(wds)
        std_wd = np.std(wds)
        for m in members:
            result[m] = (
                (within_degree[m] - mean_wd) / std_wd if std_wd > 0 else 0.0
            )
    return result
