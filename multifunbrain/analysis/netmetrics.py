"""Standard network analysis metrics for unsigned brain graphs.

Section 3 support: these metrics are applied *after* the correlation matrix
has been filtered into an unsigned weighted network (via ``filtering.py``).

Provides global graph metrics, node-level centrality measures, community
detection wrappers, degree-distribution analysis, and rich-club analysis.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "compute_global_metrics",
    "compute_node_metrics",
    "detect_communities_louvain",
    "detect_communities_spectral",
    "degree_distribution_analysis",
    "compute_rich_club_curve",
    "network_summary_report",
]


# ──────────────────────────────────────────────────────────────────────
# 1. Global metrics
# ──────────────────────────────────────────────────────────────────────


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
        Global network statistics.
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


# ──────────────────────────────────────────────────────────────────────
# 2. Node-level metrics
# ──────────────────────────────────────────────────────────────────────


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
        Mapping ``{node: community_id}``.  If provided, also computes
        participation coefficient and within-module degree z-score.

    Returns
    -------
    pd.DataFrame
        Indexed by node.
    """
    nodes = list(G.nodes())
    data: dict[str, dict] = {}

    # Degree (unweighted)
    degree = dict(G.degree())
    data["degree"] = degree

    # Strength (weighted degree)
    strength = dict(G.degree(weight=weight))
    data["strength"] = strength

    # Clustering
    data["clustering"] = nx.clustering(G, weight=weight)

    # Betweenness centrality
    data["betweenness"] = nx.betweenness_centrality(G, weight=weight)

    # Closeness centrality
    data["closeness"] = nx.closeness_centrality(G, distance=weight)

    # Eigenvector centrality
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
    """Participation coefficient P_i (Guimera & Amaral 2005).

    ``P_i = 1 - sum_s (k_is / k_i)^2``

    where *k_is* is the strength of node *i* to community *s* and *k_i*
    is the total strength.
    """
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
    """Within-module degree z-score (Guimera & Amaral 2005).

    ``z_i = (k_i_within - mean) / std``

    where the mean and std are computed over all nodes in the same module.
    """
    # Group nodes by community
    communities: dict[int, list[int]] = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    # Compute within-module degree for each node
    within_degree: dict[int, float] = {}
    for node in G.nodes():
        comm = partition.get(node)
        wd = 0.0
        for neighbour in G.neighbors(node):
            if partition.get(neighbour) == comm:
                wd += G[node][neighbour].get(weight, 1.0) if weight else 1.0
        within_degree[node] = wd

    # Z-score within each module
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


# ──────────────────────────────────────────────────────────────────────
# 3. Community detection wrappers
# ──────────────────────────────────────────────────────────────────────


def detect_communities_louvain(
    G: nx.Graph,
    weight: str | None = "weight",
    resolution: float = 1.0,
    seed: int | None = None,
) -> dict[int, int]:
    """Louvain community detection.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str or None
        Edge weight attribute.
    resolution : float
        Resolution parameter for modularity optimisation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping ``{node: community_id}``.
    """
    communities = nx.community.louvain_communities(
        G, weight=weight, resolution=resolution, seed=seed
    )
    partition = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            partition[node] = idx
    return partition


def detect_communities_spectral(
    G: nx.Graph,
    n_communities: int | None = None,
    weight: str | None = "weight",
) -> dict[int, int]:
    """Spectral clustering on the graph Laplacian.

    If *n_communities* is ``None``, it is estimated from the eigengap
    heuristic (largest gap in the Laplacian spectrum).

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    n_communities : int or None
        Number of communities.
    weight : str or None
        Edge weight attribute.

    Returns
    -------
    dict
        Mapping ``{node: community_id}``.
    """
    from sklearn.cluster import KMeans

    from .lrglib import graph_laplacian_and_spectrum

    L, spectrum = graph_laplacian_and_spectrum(G, weight=weight, normalized=True)
    if len(spectrum) < 2:
        return {n: 0 for n in G.nodes()}

    if n_communities is None:
        # Eigengap heuristic: largest gap in sorted eigenvalues
        gaps = np.diff(spectrum)
        # Skip the trivial zero eigenvalue gap
        n_communities = int(np.argmax(gaps[1:]) + 2)
        n_communities = max(2, min(n_communities, len(spectrum) // 2))

    _, eigvecs = np.linalg.eigh(L)
    embedding = eigvecs[:, :n_communities]

    km = KMeans(n_clusters=n_communities, n_init=10, random_state=0)
    labels = km.fit_predict(embedding)

    nodes = list(G.nodes())
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}


# ──────────────────────────────────────────────────────────────────────
# 4. Degree distribution analysis
# ──────────────────────────────────────────────────────────────────────


def degree_distribution_analysis(
    G: nx.Graph,
    weight: str | None = None,
    n_bins: int = 30,
) -> dict[str, Any]:
    """Analyse the degree (or strength) distribution of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str or None
        If set, analyses strength (weighted degree) instead.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Distribution statistics and histogram.
    """
    from scipy.stats import kurtosis, skew

    if weight:
        values = np.array([d for _, d in G.degree(weight=weight)])
    else:
        values = np.array([d for _, d in G.degree()])

    hist_counts, hist_edges = np.histogram(values, bins=n_bins)

    return {
        "values": values,
        "histogram": (hist_counts, hist_edges),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skewness": float(skew(values)) if len(values) > 1 else 0.0,
        "kurtosis": float(kurtosis(values)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Rich-club analysis
# ──────────────────────────────────────────────────────────────────────


def compute_rich_club_curve(
    G: nx.Graph,
    normalized: bool = True,
    n_rand: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute the rich-club coefficient curve.

    Parameters
    ----------
    G : nx.Graph
        Input graph (unweighted for standard rich-club).
    normalized : bool
        If True, normalise against degree-preserving random graphs.
    n_rand : int
        Number of random realisations for normalisation.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        ``'k'`` (degree thresholds), ``'phi'`` (raw rich-club),
        ``'phi_rand'`` (mean random, if normalised), ``'phi_norm'``
        (normalised coefficients, if requested).
    """
    # Rich-club on unweighted version
    G_uw = nx.Graph()
    G_uw.add_nodes_from(G.nodes())
    G_uw.add_edges_from(G.edges())

    rc = nx.rich_club_coefficient(G_uw, normalized=False)
    if not rc:
        return {"k": np.array([]), "phi": np.array([])}

    ks = np.array(sorted(rc.keys()))
    phi = np.array([rc[k] for k in ks])

    result: dict[str, Any] = {"k": ks, "phi": phi}

    if normalized and len(ks) > 0:
        rng = np.random.default_rng(seed)
        phi_rand_sum = np.zeros_like(phi, dtype=float)

        for _ in range(n_rand):
            G_rand = _degree_preserving_randomisation(G_uw, rng)
            rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
            for idx, k in enumerate(ks):
                phi_rand_sum[idx] += rc_rand.get(k, 0.0)

        phi_rand = phi_rand_sum / n_rand
        phi_norm = np.divide(
            phi, phi_rand, out=np.ones_like(phi, dtype=float), where=phi_rand > 0
        )
        result["phi_rand"] = phi_rand
        result["phi_norm"] = phi_norm

    return result


def _degree_preserving_randomisation(
    G: nx.Graph,
    rng: np.random.Generator,
    n_swaps: int | None = None,
) -> nx.Graph:
    """Degree-preserving edge randomisation via double-edge swaps."""
    G_rand = G.copy()
    if n_swaps is None:
        n_swaps = G.number_of_edges() * 10
    try:
        nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 10, seed=int(rng.integers(2**31)))
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass  # best-effort
    return G_rand


# ──────────────────────────────────────────────────────────────────────
# 6. Summary report
# ──────────────────────────────────────────────────────────────────────


def network_summary_report(
    G: nx.Graph,
    weight: str | None = "weight",
    run_community_detection: bool = True,
    run_rich_club: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Produce a comprehensive summary report for a network.

    Orchestrates :func:`compute_global_metrics`, :func:`compute_node_metrics`,
    community detection, and optionally rich-club analysis.

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
