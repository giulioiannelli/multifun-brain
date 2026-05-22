"""Community detection: Louvain modularity and spectral clustering."""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = ["detect_communities_louvain", "detect_communities_spectral"]


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

    from ..lrg.kernel import graph_laplacian_and_spectrum

    L, spectrum = graph_laplacian_and_spectrum(G, weight=weight, normalized=True)
    if len(spectrum) < 2:
        return {n: 0 for n in G.nodes()}

    if n_communities is None:
        gaps = np.diff(spectrum)
        n_communities = int(np.argmax(gaps[1:]) + 2)
        n_communities = max(2, min(n_communities, len(spectrum) // 2))

    _, eigvecs = np.linalg.eigh(L)
    embedding = eigvecs[:, :n_communities]

    km = KMeans(n_clusters=n_communities, n_init=10, random_state=0)
    labels = km.fit_predict(embedding)

    nodes = list(G.nodes())
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}
