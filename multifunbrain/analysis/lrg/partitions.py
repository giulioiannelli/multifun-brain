"""LRG-driven hierarchical partitions of a correlation matrix.

Run diffusion-based clustering across a set of diffusion timescales,
yielding a multiscale hierarchy of partitions. Also includes helpers to
track which nodes switch communities across tau scales.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster

from ...preprocessing.prepare import prepare_correlation_matrix
from ..graphutils import compute_normalized_linkage
from .distance import symmetrized_inverse_distance
from .kernel import graph_laplacian_and_spectrum, rho_matrix

__all__ = [
    "compute_optimal_threshold",
    "prepared_graph_laplacian_spectrum",
    "hierarchical_partitions_from_corr",
    "linkage_at_tau_min",
    "identify_switching_nodes",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    "partition_stability_index",
    "local_partition_stability_index",
    "partition_flow_table",
]


def linkage_at_tau_min(
    graph: nx.Graph, *, normalized: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """LRG diffusion-distance linkage of a graph at ``tau_min = 1 / lambda_max``.

    ``tau_min`` (the reciprocal of the largest non-zero Laplacian eigenvalue) is
    the finest informative diffusion scale — the smallest ``tau`` at which the
    diffusion kernel still resolves the whole graph. Building the ultrametric
    linkage there gives the most fully-resolved hierarchy, which is what the
    dendrogram-comparison metrics (Baker's gamma, cophenetic shift) contrast
    between two networks.

    This is the single canonical home for the helper; ``scripts/april`` and the
    dashboard both import it from here rather than re-deriving the chain.

    Parameters
    ----------
    graph : nx.Graph
        Weighted (edge attribute ``"weight"``) undirected graph — e.g. a LANS
        backbone.
    normalized : bool, default True
        Use the symmetric-normalised Laplacian (matches the LRG hierarchy path).

    Returns
    -------
    Z : np.ndarray
        SciPy linkage matrix.
    leaf_atlas_idx : np.ndarray of int
        The graph-node id (atlas index) of each linkage leaf, in leaf order.
    tau : float
        ``1 / lambda_max``.

    Raises
    ------
    RuntimeError
        If the Laplacian has no non-zero eigenvalue (empty / edgeless graph).
    """
    from ..graphutils import compute_normalized_linkage

    node_list = list(graph.nodes())
    L, evals = graph_laplacian_and_spectrum(
        graph, weight="weight", normalized=normalized
    )
    nonzero = evals[np.abs(evals) > 1e-9]
    if nonzero.size == 0:
        raise RuntimeError("Laplacian has no non-zero eigenvalues")
    tau = 1.0 / float(nonzero.max())
    dists_cond = symmetrized_inverse_distance(tau, lambda t: rho_matrix(t, L))
    Z, _labels, _tmax = compute_normalized_linkage(
        dists_cond, graph, labelList="numbers"
    )
    return Z, np.asarray(node_list, dtype=int), tau


def compute_optimal_threshold(linkage_matrix, scaling_factor=1):
    """Optimal flat-clustering threshold from a linkage matrix via the
    partition stability index.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering. The third column
        contains the merge distances.
    scaling_factor : float, optional
        Factor applied to the optimal threshold (default 1).

    Returns
    -------
    FlatClusteringTh : float
        Threshold for flat clustering (after scaling).
    optimal_threshold : float
        Optimal threshold from dendrogram-gap analysis.
    stability_indices : np.ndarray
        Stability index per branch.
    optimal_branch_index : int
        Index of the most-stable branch.
    """
    dendro_thresholds = linkage_matrix[:, 2]
    D_values = dendro_thresholds[::-1]

    N = 1 / (np.log10(D_values[0]) - np.log10(D_values[-1]))

    stability_indices = []
    for i in range(len(D_values) - 1):
        sigma = N * (np.log10(D_values[i]) - np.log10(D_values[i + 1]))
        stability_indices.append(sigma)
    stability_indices = np.array(stability_indices)

    optimal_branch_index = np.argmax(stability_indices)
    optimal_threshold = D_values[optimal_branch_index + 1]

    FlatClusteringTh = optimal_threshold * scaling_factor

    return FlatClusteringTh, optimal_threshold, stability_indices, optimal_branch_index


def prepared_graph_laplacian_spectrum(
    corr, edge_threshold: float = 0.0, normalized: bool = True
):
    """``(graph, L, spectrum)`` from a correlation / adjacency matrix.

    The single prepare -> graph -> Laplacian chain the LRG hierarchy is built on:
    ``prepare_correlation_matrix`` drops dead regions and zeroes the diagonal, an
    optional ``edge_threshold`` sparsifies, then the (symmetric-normalised)
    Laplacian and its ascending spectrum are returned. Shared so callers that need
    only the spectrum — e.g. the dashboard's specific-heat ``C(tau)`` curve — get
    exactly the same Laplacian as the clustering path, without duplicating steps.
    """
    cleaned = prepare_correlation_matrix(corr)
    if edge_threshold > 0:
        cleaned = np.where(cleaned >= edge_threshold, cleaned, 0.0)
    graph = nx.from_numpy_array(cleaned)
    L, spectrum = graph_laplacian_and_spectrum(
        graph, weight="weight", normalized=normalized
    )
    return graph, L, spectrum


def hierarchical_partitions_from_corr(
    corr,
    tau_values: Sequence[float],
    edge_threshold: float = 0.0,
    normalized_laplacian: bool = True,
) -> list[dict]:
    """Run the diffusion / LRG-style clustering pipeline for a correlation matrix.

    Parameters
    ----------
    corr : array-like
        Square correlation matrix.
    tau_values : sequence of float
        Diffusion time scales to evaluate.
    edge_threshold : float
        Minimum edge weight; values below are set to zero.
    normalized_laplacian : bool
        Whether to use the symmetric normalised Laplacian.

    Returns
    -------
    list of dict
        One dict per *tau* value containing ``'tau'``, ``'partition'``,
        ``'linkage_matrix'``, ``'linkage_labels'``, ``'tmax'``,
        ``'flat_threshold'``.
    """
    graph, L, _spectrum = prepared_graph_laplacian_spectrum(
        corr, edge_threshold=edge_threshold, normalized=normalized_laplacian
    )

    partitions: list[dict] = []
    for tau in tau_values:
        dists = symmetrized_inverse_distance(tau, lambda t: rho_matrix(t, L))
        linkage_matrix, labels, tmax = compute_normalized_linkage(
            dists, graph, labelList="numbers"
        )
        flat_threshold, _, _, _ = compute_optimal_threshold(linkage_matrix)
        partition = fcluster(linkage_matrix, flat_threshold, criterion="distance")
        partitions.append(
            {
                "tau": float(tau),
                "partition": partition,
                "linkage_matrix": linkage_matrix,
                "linkage_labels": labels,
                "tmax": tmax,
                "flat_threshold": flat_threshold,
            }
        )
    return partitions


def identify_switching_nodes(partitions, tau_values):
    """Nodes whose community assignment changes across tau scales.

    Parameters
    ----------
    partitions : list of array-like
        Community labels per node at each tau scale.
    tau_values : sequence of float
        Tau values corresponding to each partition.

    Returns
    -------
    dict
        ``{node_index: [(tau, community), ...]}`` for nodes whose
        assignment is not constant across all tau values.
    """
    if not partitions:
        return {}

    n_nodes = len(partitions[0])
    result = {}
    for node in range(n_nodes):
        history = [(float(tau_values[i]), int(partitions[i][node])) for i in range(len(tau_values))]
        if len({assignment for _, assignment in history}) > 1:
            result[node] = history
    return result


def get_moved_nodes(partdict_tau, source_cluster):
    """First-move event for nodes leaving ``source_cluster`` as tau grows.

    Parameters
    ----------
    partdict_tau : dict
        ``{node_id: [(tau, cluster), ...]}``.
    source_cluster : int
        Initial cluster of interest.

    Returns
    -------
    dict
        ``{node_id: {'tau_move', 'new_cluster', 'history'}}`` for nodes
        that start in *source_cluster* and later move out.
    """
    moved = {}
    for node, history in partdict_tau.items():
        if history[0][1] == source_cluster:
            for tau, cluster in history:
                if cluster != source_cluster:
                    moved[node] = {"tau_move": tau, "new_cluster": cluster, "history": history}
                    break
    return moved


def partition_stability_index(linkage_matrix: np.ndarray) -> np.ndarray:
    """Full ``Ψ(n)`` array from a linkage matrix.

    Implements Eq. (2) of Villegas et al., *Phys. Rev. Research* 7, 013065
    (2025): ``Ψ(n) = N · [log10 Δ_n − log10 Δ_{n+1}]`` where ``Δ_n`` is the
    threshold of the n-th branching when the dendrogram is read from the top
    (so ``Δ`` is decreasing in n), and ``N`` is the inverse of the full
    log-range, ``N = 1 / [log10 Δ_1 − log10 Δ_{n_max}]``.

    The optimal cut is ``argmax Ψ(n)``; multiple local maxima correspond to
    multiple characteristic mesoscopic scales.

    Parameters
    ----------
    linkage_matrix : np.ndarray, shape ``(N-1, 4)``
        Linkage matrix from ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    np.ndarray, shape ``(N-2,)``
        ``Ψ(n)`` for ``n = 1, ..., N-2``.
    """
    dendro_thresholds = linkage_matrix[:, 2]
    D_values = dendro_thresholds[::-1]
    positive = D_values > 0
    if not positive.all():
        floor = D_values[positive].min() if positive.any() else 1e-12
        D_values = np.maximum(D_values, floor)
    log_D = np.log10(D_values)
    norm = 1.0 / (log_D[0] - log_D[-1])
    return norm * (log_D[:-1] - log_D[1:])


def local_partition_stability_index(
    linkage_matrix: np.ndarray,
    cluster_member_indices: Sequence[int],
) -> float:
    """Local partition stability ``Ψ_L`` for one cluster within a dendrogram.

    Per Villegas et al. 2025, ``Ψ_L(τ)`` measures the log-range of the
    dendrogram branch between the **creation** of a cluster (the merge that
    fuses its two children) and its **dissolution** (the merge that absorbs
    it into a supercluster), normalised by the full dendrogram log-range.
    High ``Ψ_L`` ⇒ the cluster is locally stable over a wide range of
    cut thresholds.

    Parameters
    ----------
    linkage_matrix : np.ndarray, shape ``(N-1, 4)``
    cluster_member_indices : sequence of int
        Original leaf indices (0-based) of the candidate cluster.

    Returns
    -------
    float
        ``Ψ_L`` for the given cluster, or ``NaN`` if the cluster does not
        correspond to any internal node of the dendrogram.
    """
    n_leaves = linkage_matrix.shape[0] + 1
    members = frozenset(int(i) for i in cluster_member_indices)

    leaves_of: dict[int, frozenset[int]] = {i: frozenset({i}) for i in range(n_leaves)}
    for k in range(n_leaves - 1):
        a = int(linkage_matrix[k, 0])
        b = int(linkage_matrix[k, 1])
        leaves_of[n_leaves + k] = leaves_of[a] | leaves_of[b]

    creation_id = None
    creation_height = None
    for k in range(n_leaves - 1):
        cid = n_leaves + k
        if leaves_of[cid] == members:
            creation_height = float(linkage_matrix[k, 2])
            creation_id = cid
            break
    if creation_id is None:
        return float("nan")

    dissolution_height = None
    for k in range(n_leaves - 1):
        a = int(linkage_matrix[k, 0])
        b = int(linkage_matrix[k, 1])
        if a == creation_id or b == creation_id:
            dissolution_height = float(linkage_matrix[k, 2])
            break
    if dissolution_height is None:
        return float("nan")

    heights = linkage_matrix[:, 2]
    positive = heights > 0
    floor = heights[positive].min() if positive.any() else 1e-12
    log_max = np.log10(max(heights.max(), floor))
    log_min = np.log10(floor)
    norm = 1.0 / (log_max - log_min) if log_max > log_min else 0.0

    creation_height = max(creation_height, floor)
    dissolution_height = max(dissolution_height, floor)
    return float(norm * (np.log10(dissolution_height) - np.log10(creation_height)))


def partition_flow_table(per_tau_results: list[dict]):
    """Sankey-ready node migration table across consecutive ``tau`` scales.

    Parameters
    ----------
    per_tau_results : list of dict
        Each entry must carry ``'tau'`` and ``'partition'`` (1D label array
        of length ``N``).

    Returns
    -------
    pandas.DataFrame
        Columns ``['tau_from', 'tau_to', 'node', 'cluster_from', 'cluster_to']``;
        one row per node per consecutive ``tau`` step.
    """
    import pandas as pd

    rows = []
    for k in range(len(per_tau_results) - 1):
        a = per_tau_results[k]
        b = per_tau_results[k + 1]
        part_a = np.asarray(a["partition"])
        part_b = np.asarray(b["partition"])
        for node in range(len(part_a)):
            rows.append({
                "tau_from": float(a["tau"]),
                "tau_to": float(b["tau"]),
                "node": int(node),
                "cluster_from": int(part_a[node]),
                "cluster_to": int(part_b[node]),
            })
    return pd.DataFrame(rows)


def get_moved_nodes_interval(partdict_tau, source_cluster, tau_i=None, tau_f=None, tol=1e-3):
    """Nodes that move between two specific tau values.

    Parameters
    ----------
    partdict_tau : dict
        ``{node_id: [(tau, cluster), ...]}``.
    source_cluster : int
        Cluster at ``tau_i``.
    tau_i, tau_f : float or None
        Start and end tau values. If ``None``, use the first two tau
        entries of an arbitrary node's history.
    tol : float
        Tolerance for matching tau values.

    Returns
    -------
    dict
        ``{node_id: {'tau_i_cluster', 'tau_f_cluster', 'history'}}`` for
        nodes that started in *source_cluster* at ``tau_i`` and changed
        cluster by ``tau_f``.
    """
    if tau_i is None or tau_f is None:
        sample_node = next(iter(partdict_tau))
        sample_history = partdict_tau[sample_node]
        if len(sample_history) >= 2:
            if tau_i is None:
                tau_i = sample_history[0][0]
            if tau_f is None:
                tau_f = sample_history[1][0]
        else:
            raise ValueError("Not enough tau values in node history to set defaults.")

    moved = {}
    for node, history in partdict_tau.items():
        cluster_i = None
        cluster_f = None
        for t, cluster in history:
            if abs(t - tau_i) < tol:
                cluster_i = cluster
            if abs(t - tau_f) < tol:
                cluster_f = cluster
        if cluster_i is None or cluster_f is None:
            continue
        if cluster_i == source_cluster and cluster_f != source_cluster:
            moved[node] = {
                "tau_i_cluster": cluster_i,
                "tau_f_cluster": cluster_f,
                "history": history,
            }
    return moved
