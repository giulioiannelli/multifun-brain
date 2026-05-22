"""Shared helper: build a graph from a non-negative adjacency matrix and
extract its giant component.

Used internally by :mod:`multifunbrain.processing.filtering` and
:mod:`multifunbrain.processing.backbone`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..analysis.graphutils import get_giant_component_leftoff

__all__ = ["matrix_to_graph_giant"]


def matrix_to_graph_giant(matrix: np.ndarray) -> tuple[nx.Graph, list]:
    """Build a graph from a non-negative adjacency matrix and return its
    giant connected component.

    Parameters
    ----------
    matrix : np.ndarray
        Square non-negative adjacency matrix (diagonal will be zeroed).

    Returns
    -------
    G : nx.Graph
        Giant connected component (or the input if already small).
    removed : list
        Nodes not in the giant component.
    """
    np.fill_diagonal(matrix, 0.0)
    G_full = nx.from_numpy_array(matrix)
    zero_edges = [(u, v) for u, v, w in G_full.edges(data="weight") if w == 0]
    G_full.remove_edges_from(zero_edges)

    if G_full.number_of_edges() == 0:
        return G_full, list(G_full.nodes())

    G, removed = get_giant_component_leftoff(G_full, remove_selfloops=True)
    return G, removed
