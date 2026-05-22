"""Strength-preserving null models for weighted networks.

The headline routine is :func:`strength_preserving_rewire` — a
weighted-network equivalent of the Maslov–Sneppen double-edge swap
that approximately preserves each node's strength (sum of incident
edge weights) while destroying community structure. This is the
correct null for hypotheses about *diffusion through edges* (LRG
hierarchy), since both node strengths and overall weight distribution
shape the Laplacian spectrum.

Algorithm — Rubinov & Sporns 2011, *NeuroImage*:

1. Randomise topology with :func:`networkx.double_edge_swap` on the
   binarised graph, so each node's binary degree is preserved exactly.
2. Sort the original edge weights ascending; rank the rewired edges by
   the product of their endpoints' **target** strengths. Assign sorted
   weights in matching rank order (high strength × high strength gets
   the heaviest weight).
3. Iteratively swap weights between pairs of rewired edges (no shared
   endpoints) whenever the swap reduces the per-node squared strength
   error. ``weight_iters`` controls how many candidate swaps are
   attempted; ~100 N gives < 5 % mean strength error on typical LANS
   backbones.

The cached batch helper :func:`cached_surrogate_linkages` builds
LRG dendrograms on a set of strength-preserving surrogates and caches
the result as a pickle. It is **resumable** — calling it with a
partial cache extends the cache to the requested ``n_surrogates`` and
re-writes only the appended entries.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "strength_preserving_rewire",
    "cached_surrogate_linkages",
]


def strength_preserving_rewire(
    G: nx.Graph,
    rng: np.random.Generator,
    *,
    n_swaps: int | None = None,
    weight_iters: int | None = None,
) -> nx.Graph:
    """Strength-preserving rewiring of a weighted undirected graph.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected graph (edge attribute ``"weight"``). Self
        loops are stripped silently before processing.
    rng : np.random.Generator
        Source of randomness. Pass ``np.random.default_rng(seed)`` for
        deterministic output.
    n_swaps : int, optional
        Number of double-edge swap attempts for the topology phase.
        Default: ``10 * |E|``.
    weight_iters : int, optional
        Number of random pairwise weight-swap attempts in the
        strength-refinement phase. Default: ``100 * |E|``.

    Returns
    -------
    nx.Graph
        New graph with the same node set as ``G``, rewired edges, and
        edge weights chosen so each node's strength approximates its
        original value.
    """
    G_clean = G.copy()
    G_clean.remove_edges_from(list(nx.selfloop_edges(G_clean)))
    if G_clean.number_of_edges() == 0:
        return G_clean

    nodes = list(G_clean.nodes())
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    target_strength = np.zeros(len(nodes), dtype=float)
    for u, v, d in G_clean.edges(data=True):
        w = float(d.get("weight", 1.0))
        target_strength[node_idx[u]] += w
        target_strength[node_idx[v]] += w

    G_topo = nx.Graph()
    G_topo.add_nodes_from(nodes)
    G_topo.add_edges_from(G_clean.edges())
    n_edges = G_topo.number_of_edges()

    if n_swaps is None:
        n_swaps = max(1, 10 * n_edges)
    if weight_iters is None:
        weight_iters = max(50, 100 * n_edges)

    if n_edges >= 2:
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            nx.double_edge_swap(
                G_topo,
                nswap=n_swaps,
                max_tries=max(100, n_swaps * 10),
                seed=seed,
            )
        except (nx.NetworkXAlgorithmError, nx.NetworkXError) as exc:
            log.debug("double_edge_swap skipped: %s", exc)

    edges = list(G_topo.edges())
    sorted_weights = np.sort(
        np.array(
            [float(d.get("weight", 1.0)) for _, _, d in G_clean.edges(data=True)],
            dtype=float,
        )
    )

    products = np.array(
        [target_strength[node_idx[u]] * target_strength[node_idx[v]]
         for u, v in edges],
        dtype=float,
    )
    edge_order = np.argsort(products, kind="stable")
    cur_w = np.empty(n_edges, dtype=float)
    cur_w[edge_order] = sorted_weights

    cur_strength = np.zeros(len(nodes), dtype=float)
    edges_u = np.array([node_idx[u] for u, _ in edges], dtype=int)
    edges_v = np.array([node_idx[v] for _, v in edges], dtype=int)
    np.add.at(cur_strength, edges_u, cur_w)
    np.add.at(cur_strength, edges_v, cur_w)

    if weight_iters > 0 and n_edges >= 2:
        i_arr = rng.integers(0, n_edges, size=weight_iters)
        j_arr = rng.integers(0, n_edges, size=weight_iters)
        for k in range(weight_iters):
            i = int(i_arr[k])
            j = int(j_arr[k])
            if i == j:
                continue
            ui, vi = edges_u[i], edges_v[i]
            uj, vj = edges_u[j], edges_v[j]
            if ui == uj or ui == vj or vi == uj or vi == vj:
                continue
            wi, wj = cur_w[i], cur_w[j]
            if wi == wj:
                continue
            delta = wj - wi
            err_ui_before = cur_strength[ui] - target_strength[ui]
            err_vi_before = cur_strength[vi] - target_strength[vi]
            err_uj_before = cur_strength[uj] - target_strength[uj]
            err_vj_before = cur_strength[vj] - target_strength[vj]
            err_before = (
                err_ui_before * err_ui_before
                + err_vi_before * err_vi_before
                + err_uj_before * err_uj_before
                + err_vj_before * err_vj_before
            )
            err_ui_after = err_ui_before + delta
            err_vi_after = err_vi_before + delta
            err_uj_after = err_uj_before - delta
            err_vj_after = err_vj_before - delta
            err_after = (
                err_ui_after * err_ui_after
                + err_vi_after * err_vi_after
                + err_uj_after * err_uj_after
                + err_vj_after * err_vj_after
            )
            if err_after < err_before:
                cur_w[i], cur_w[j] = wj, wi
                cur_strength[ui] += delta
                cur_strength[vi] += delta
                cur_strength[uj] -= delta
                cur_strength[vj] -= delta

    G_out = nx.Graph()
    G_out.add_nodes_from(nodes)
    for (u, v), w in zip(edges, cur_w):
        G_out.add_edge(u, v, weight=float(w))
    return G_out


def cached_surrogate_linkages(
    G: nx.Graph,
    label: str,
    n_surrogates: int,
    cache_dir: Path,
    *,
    linkage_fn: Callable[[nx.Graph], tuple[np.ndarray, np.ndarray, float]],
    rng_seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate (or resume) cached LRG linkages on strength-preserving surrogates.

    Each cache entry is a dict ``{"linkage": Z, "leaf_atlas_idx": nodes, "tau": tau}``
    where ``Z`` is the scipy linkage matrix on the surrogate graph,
    ``nodes`` is the atlas-index of each leaf, and ``tau`` is the
    τ_min = 1/λ_max used to build the linkage.

    Parameters
    ----------
    G : nx.Graph
        LANS backbone of the source matrix. Surrogates are rewired
        versions of this graph.
    label : str
        Unique key for the source matrix (used as cache file stem).
    n_surrogates : int
        Total surrogates wanted in the cache after this call. If the
        cache already has more, the first ``n_surrogates`` are returned.
        If fewer, the missing ones are computed and appended.
    cache_dir : Path
        Directory where ``<label>__surrogates.pkl`` lives.
    linkage_fn : callable
        Builds ``(Z, leaf_atlas_idx, tau)`` from a graph. Pass the
        ``linkage_at_tau_min`` helper from ``scripts/april/_fig_common.py``
        (or any equivalent).
    rng_seed : int, default 0
        Seed for the strength-preserving rewiring RNG. Surrogate ``i``
        uses ``np.random.default_rng(rng_seed + i)``.

    Returns
    -------
    list of dict
        ``n_surrogates`` entries, each ``{"linkage", "leaf_atlas_idx", "tau"}``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{label}__surrogates.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            existing = pickle.load(f)
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []

    if len(existing) >= n_surrogates:
        return existing[:n_surrogates]

    to_compute = n_surrogates - len(existing)
    log.info(
        "[%s] surrogate cache: have %d / need %d, computing %d more",
        label, len(existing), n_surrogates, to_compute,
    )
    for i in range(len(existing), n_surrogates):
        rng = np.random.default_rng(rng_seed + i)
        G_sur = strength_preserving_rewire(G, rng)
        if G_sur.number_of_nodes() < 4 or G_sur.number_of_edges() < 3:
            continue
        try:
            Z, leaf_idx, tau = linkage_fn(G_sur)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[%s] surrogate %d linkage failed: %s", label, i, exc)
            continue
        existing.append(
            {
                "linkage": Z,
                "leaf_atlas_idx": np.asarray(leaf_idx, dtype=int),
                "tau": float(tau),
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == n_surrogates:
            with open(cache_path, "wb") as f:
                pickle.dump(existing, f)
    with open(cache_path, "wb") as f:
        pickle.dump(existing, f)

    return existing[:n_surrogates]
