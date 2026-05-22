"""Global graph-level metrics for unsigned weighted networks."""

from __future__ import annotations

from typing import Any

import networkx as nx

__all__ = ["compute_global_metrics"]


def compute_global_metrics(
    G: nx.Graph,
    weight: str | None = "weight",
) -> dict[str, Any]:
    """Compute standard global graph metrics.

    Parameters
    ----------
    G : nx.Graph
        Input graph (undirected, optionally weighted).
    weight : str or None
        Edge attribute for weighted computations.

    Returns
    -------
    dict
        Global network statistics: ``n_nodes``, ``n_edges``, ``density``,
        ``is_connected``, ``n_components``, ``avg_clustering``,
        ``transitivity``, ``avg_shortest_path``, ``diameter``,
        ``global_efficiency``, ``assortativity``.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    connected = nx.is_connected(G) if n > 0 else False
    n_comp = nx.number_connected_components(G) if n > 0 else 0

    metrics: dict[str, Any] = {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
        "is_connected": connected,
        "n_components": n_comp,
    }

    if n == 0:
        for k in (
            "avg_clustering",
            "transitivity",
            "avg_shortest_path",
            "diameter",
            "global_efficiency",
            "assortativity",
        ):
            metrics[k] = float("nan")
        return metrics

    metrics["avg_clustering"] = nx.average_clustering(G, weight=weight)
    metrics["transitivity"] = nx.transitivity(G)

    if connected and n > 1:
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(
            G, weight=weight
        )
        metrics["diameter"] = nx.diameter(G)
    else:
        metrics["avg_shortest_path"] = float("nan")
        metrics["diameter"] = float("nan")

    metrics["global_efficiency"] = nx.global_efficiency(G)

    try:
        metrics["assortativity"] = nx.degree_assortativity_coefficient(
            G, weight=weight
        )
    except (ValueError, ZeroDivisionError):
        metrics["assortativity"] = float("nan")

    return metrics
