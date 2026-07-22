"""Tests for the LRG diffusion-distance 2-D layout (``analysis.lrg.layout``)."""

from __future__ import annotations

import networkx as nx
import numpy as np

from multifunbrain.analysis.lrg.layout import (
    diffusion_distance_matrix,
    diffusion_distance_mds,
)


def _block_graph(seed: int = 0) -> nx.Graph:
    """A 3-block weighted graph with strong intra-block, weak inter-block edges."""
    r = np.random.default_rng(seed)
    n_per, k = 8, 3
    n = n_per * k
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i // n_per) == (j // n_per)
            A[i, j] = A[j, i] = abs(r.normal(0.7 if same else 0.08, 0.03))
    return nx.from_numpy_array(A)


def test_distance_matrix_is_symmetric_zero_diagonal():
    g = _block_graph()
    tau = 1.0
    d = diffusion_distance_matrix(g, tau)
    assert d.shape == (g.number_of_nodes(), g.number_of_nodes())
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert np.isfinite(d).all()


def test_mds_shape_and_finite():
    g = _block_graph()
    coords = diffusion_distance_mds(g, 1.0)
    assert coords.shape == (g.number_of_nodes(), 2)
    assert np.isfinite(coords).all()
    # Normalised into a [-1, 1] box.
    assert np.abs(coords).max() <= 1.0 + 1e-9


def test_mds_node_order_matches_graph():
    g = _block_graph()
    coords = diffusion_distance_mds(g, 1.0)
    assert coords.shape[0] == len(list(g.nodes()))


def test_mds_places_blocks_together():
    # Intra-block MDS distances should be smaller than inter-block on average,
    # i.e. the embedding respects the diffusion communities.
    g = _block_graph(seed=1)
    coords = diffusion_distance_mds(g, 1.0)
    n_per = 8
    from scipy.spatial.distance import cdist

    d = cdist(coords, coords)
    intra, inter = [], []
    n = coords.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            (intra if i // n_per == j // n_per else inter).append(d[i, j])
    assert np.mean(intra) < np.mean(inter)


def test_mds_degenerate_small_graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    coords = diffusion_distance_mds(g, 1.0)
    assert coords.shape == (2, 2)
    assert np.isfinite(coords).all()


def _two_triangles() -> nx.Graph:
    g = nx.Graph()
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        g.add_edge(a, b, weight=1.0)
    return g  # two disconnected components {0,1,2}, {3,4,5}


def test_disconnected_cross_component_distance_is_large_not_zero():
    # A disconnected graph must NOT map cross-component (infinite) diffusion
    # distance to 0 — that would place the two components on top of each other.
    g = _two_triangles()
    d = diffusion_distance_matrix(g, 1.0)
    assert np.isfinite(d).all()
    within = d[0, 1]
    cross = d[0, 3]
    assert cross > within  # unreachable-by-diffusion pairs are the farthest apart


def test_disconnected_components_are_separated_by_mds():
    g = _two_triangles()
    coords = diffusion_distance_mds(g, 1.0)
    assert np.isfinite(coords).all()
    ca = coords[:3].mean(axis=0)
    cb = coords[3:].mean(axis=0)
    intra = np.linalg.norm(coords[:3] - ca, axis=1).mean()
    sep = float(np.linalg.norm(ca - cb))
    # Components are pushed clearly apart, not collapsed onto one another.
    assert sep > intra
