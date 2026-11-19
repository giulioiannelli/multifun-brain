"""Serializers for the LRG multiscale section (per filter) of a PipelineResult.

A result's ``lrg_results[filter]`` is a list of per-tau dicts
(``tau``, ``partition``, ``linkage_matrix``, ``flat_threshold``, ``tmax``) or an
error dict. From these we derive an interactive dendrogram (SciPy layout, coloured
client-side by a moving cut), a partition-flow raster across tau, a community-flow
Sankey, the partition-stability ``Psi(n)`` curve (dendrogram gap at one tau), and
the specific heat ``C(tau)`` (recomputed online from the filtered graph's Laplacian
spectrum — the exact Laplacian the partitions use, so no re-ingest is needed).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster

from multifunbrain.analysis.lrg.distance import symmetrized_inverse_distance
from multifunbrain.analysis.lrg.kernel import (
    entropy,
    graph_laplacian_and_spectrum,
    rho_matrix,
)
from multifunbrain.analysis.lrg.layout import diffusion_distance_mds
from multifunbrain.analysis.lrg.partitions import (
    compute_optimal_threshold,
    prepared_graph_laplacian_spectrum,
)
from multifunbrain.analysis.lrg.scales import characteristic_scales

from ..encode import clean
from ..graphs import derive_graph
from ..remap import surviving_labels


def resolve_filter(result, filter_name: str | None) -> str | None:
    filters = list(result.lrg_results.keys())
    if not filters:
        return None
    if filter_name in filters:
        return filter_name
    return filters[0]


def _recompute_entries(graph: nx.Graph, normalized: bool, n_steps: int = 14) -> list[dict]:
    """LRG per-τ partitions/linkages built directly on *graph*.

    Mirrors the pipeline's ``lrg_results`` format but is computed on an ad-hoc
    sparsified backbone, and keeps the graph's own node ordering (so the caller
    can map leaves back to atlas indices). The τ grid spans the finest scale
    ``1/λ_max`` to the coarsest ``1/λ_min`` (reciprocals of the extreme non-zero
    Laplacian eigenvalues) — the informative diffusion range.
    """
    from multifunbrain.analysis.graphutils import compute_normalized_linkage

    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return []
    L, spectrum = graph_laplacian_and_spectrum(graph, weight="weight", normalized=normalized)
    nz = spectrum[np.abs(spectrum) > 1e-9]
    if nz.size == 0:
        return []
    tau_min = 1.0 / float(nz.max())
    tau_max = 1.0 / float(nz.min())
    if not (tau_max > tau_min):
        tau_max = tau_min * 1e3
    taus = np.geomspace(tau_min, tau_max, n_steps)

    entries: list[dict] = []
    for tau in taus:
        dists = symmetrized_inverse_distance(tau, lambda t: rho_matrix(t, L))
        try:
            Z, _labels, tmax = compute_normalized_linkage(dists, graph, labelList="numbers")
            flat, _, _, _ = compute_optimal_threshold(Z)
            partition = fcluster(Z, flat, criterion="distance")
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue
        entries.append(
            {
                "tau": float(tau),
                "partition": np.asarray(partition),
                "linkage_matrix": np.asarray(Z, dtype=float),
                "flat_threshold": float(flat),
                "tmax": tmax,
            }
        )
    return entries


def _lrg_entries(
    result,
    filter_name,
    sparsify: str | None = None,
    alpha: float = 0.05,
    threshold: float = 0.3,
):
    """``(filter, entries, node_ids)`` honouring an ad-hoc *sparsify*.

    ``filter``/``None``/``""`` returns the pre-computed ``result.lrg_results``
    (``node_ids`` is ``None`` → leaves align to ``surviving_labels`` positions).
    Any other *sparsify* re-derives the backbone (:func:`derive_graph`) and
    recomputes the LRG hierarchy on it, returning the graph's atlas-index
    ``node_ids`` so leaves stay named. Recomputes are cached on the result.
    """
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return None, None, None

    if sparsify in (None, "", "filter"):
        parts = result.lrg_results[fname]
        if isinstance(parts, dict) or not isinstance(parts, list) or not parts:
            return fname, None, None
        # The stored linkage was built on the filter graph, whose giant component
        # may have dropped a few nodes → its leaves are that graph's node order
        # (survivor positions), not 0..n_survivors-1. Recover the mapping so leaves
        # keep atlas names / coords whenever the counts line up.
        node_ids = None
        info = result.filtered_networks.get(fname) if result.filtered_networks else None
        graph = info.get("graph") if info else None
        if graph is not None:
            nid = [int(x) for x in graph.nodes()]
            n_leaves = int(np.asarray(parts[0]["linkage_matrix"]).shape[0] + 1)
            if len(nid) == n_leaves:
                node_ids = nid
        return fname, parts, node_ids

    if not result.filtered_networks or fname not in result.filtered_networks:
        parts = result.lrg_results.get(fname)
        parts = parts if isinstance(parts, list) and parts else None
        return fname, parts, None

    cache = getattr(result, "_lrg_sparsify_cache", None)
    if cache is None:
        cache = {}
        try:
            result._lrg_sparsify_cache = cache
        except (AttributeError, TypeError):
            cache = {}
    key = (fname, sparsify, round(float(alpha), 4), round(float(threshold), 4))
    if key in cache:
        entries, node_ids = cache[key]
    else:
        graph, eff = derive_graph(result, fname, sparsify, alpha, threshold)
        if eff == "filter":
            # Backbone collapsed / unavailable → fall back to the stored hierarchy.
            parts = result.lrg_results.get(fname)
            parts = parts if isinstance(parts, list) and parts else None
            cache[key] = (parts, None)
            return fname, parts, None
        normalized = getattr(result.config, "normalized_laplacian", True)
        entries = _recompute_entries(graph, normalized)
        node_ids = [int(n) for n in graph.nodes()]
        cache[key] = (entries if entries else None, node_ids)
        entries = entries if entries else None
    return fname, entries, node_ids


def _leaf_labels(result, n_leaves: int, node_ids: list[int] | None) -> list[str]:
    """Short leaf names for a dendrogram/network with *n_leaves* leaves.

    Maps by atlas index when ``node_ids`` is given (sparsified backbones that
    dropped nodes), else assumes leaves align 1-to-1 with ``surviving_labels``.
    Falls back to ``node-i`` when neither matches.
    """
    nodes = surviving_labels(result)
    if node_ids is not None and len(node_ids) == n_leaves:
        return [
            nodes[i]["short"] if 0 <= i < len(nodes) else f"node-{i}" for i in node_ids
        ]
    if len(nodes) == n_leaves:
        return [nd["short"] for nd in nodes]
    return [f"node-{i}" for i in range(n_leaves)]


# Qualitative palette shared by the dendrogram cut, the partition-flow raster, and
# the Sankey so a community keeps one colour across all three LRG views.
_LRG_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#393b79", "#637939", "#8c6d31",
    "#843c39", "#7b4173", "#5254a3", "#e7969c", "#a55194", "#6b6ecf",
]


def _canonical_labels(partitions):
    """Relabel per-τ cluster ids for cross-τ colour continuity.

    Raw LRG cluster ids are arbitrary per τ (id 0 at one τ is unrelated to id 0 at
    the next), which makes the flow views flicker meaninglessly. Here each cluster
    at step *k* inherits the canonical id of the step *k−1* cluster it overlaps most
    (largest clusters first; a canonical id already claimed in the current column
    forces a fresh id, so a genuine split stays two colours). Returns an
    ``(n_nodes, n_steps)`` int array of canonical ids.
    """
    T = len(partitions)
    n = int(partitions[0].size) if T else 0
    canon = np.zeros((n, T), dtype=int)
    if T == 0:
        return canon, 0

    first = partitions[0].astype(int)
    remap = {c: i for i, c in enumerate(sorted(set(first.tolist())))}
    canon[:, 0] = [remap[c] for c in first.tolist()]
    next_id = len(remap)
    prev = canon[:, 0]

    for k in range(1, T):
        cur = partitions[k].astype(int)
        # Largest clusters get first pick of their predecessor's colour.
        clusters = sorted(set(cur.tolist()), key=lambda c: -int((cur == c).sum()))
        used: set[int] = set()
        assigned: dict[int, int] = {}
        for c in clusters:
            mask = cur == c
            ids, counts = np.unique(prev[mask], return_counts=True)
            chosen = None
            for j in np.argsort(-counts):
                pid = int(ids[j])
                if pid not in used:
                    chosen = pid
                    break
            if chosen is None:
                chosen = next_id
                next_id += 1
            used.add(chosen)
            assigned[c] = chosen
        canon[:, k] = [assigned[c] for c in cur.tolist()]
        prev = canon[:, k]

    return canon, next_id


def tau_grid_spec(
    result,
    *,
    filter_name: str | None = None,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """List of available tau steps (for the LRG tau selector)."""
    fname, parts, _nid = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "tau_grid", "label": result.label, "filter": fname, "taus": []}
    taus = [float(p["tau"]) for p in parts]
    nclusters = [int(np.unique(np.asarray(p["partition"])).size) for p in parts]
    return clean(
        {"kind": "tau_grid", "label": result.label, "filter": fname,
         "sparsify": sparsify or "filter", "taus": taus, "n_clusters": nclusters}
    )


def dendrogram_spec(
    result,
    *,
    filter_name: str | None = None,
    tau_index: int = -1,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """SciPy dendrogram at one τ, plus the data to recolour it at ANY cut *h*
    client-side (no refetch).

    The expensive per-τ linkage is laid out once here; the payload carries the
    linkage ``Z`` and, per drawn link, its linkage node id (``link_ids``), so the
    browser runs a union-find over merges below a moving height *h* to colour
    branches per cluster (grey above) — dragging the height slider never
    round-trips. Only changing τ (a new linkage) refetches. ``flat_threshold`` is
    the LRG's natural default cut; ``dcoord_min_positive`` / ``dcoord_max`` bound
    the slider.
    """
    fname, parts, node_ids = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "dendrogram", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    tau_index = max(-len(parts), min(tau_index, len(parts) - 1))
    entry = parts[tau_index]
    linkage = np.asarray(entry["linkage_matrix"], dtype=float)
    labels = _leaf_labels(result, linkage.shape[0] + 1, node_ids)

    # One layout pass. ``link_color_func`` tags each drawn link with its linkage
    # node id (>= n_leaves), returned in draw order as ``color_list`` — this is how
    # the client maps a drawn link back to its merge (and thus its cluster at h).
    dn = dendrogram(linkage, no_plot=True, labels=labels, link_color_func=lambda k: str(k))
    link_ids = [int(c) for c in dn["color_list"]]
    dcoord = np.asarray(dn["dcoord"], dtype=float)
    positive = dcoord[dcoord > 0]

    return clean(
        {
            "kind": "dendrogram",
            "label": result.label,
            "filter": fname,
            "sparsify": sparsify or "filter",
            "tau": float(entry["tau"]),
            "tau_index": int(tau_index if tau_index >= 0 else len(parts) + tau_index),
            "icoord": dn["icoord"],
            "dcoord": dn["dcoord"],
            "ivl": dn["ivl"],
            "leaves": dn["leaves"],
            "linkage": linkage,  # (n-1, 4) Z matrix -> client union-find
            "link_ids": link_ids,  # linkage node id (>= n_leaves) per drawn link
            "n_leaves": int(linkage.shape[0] + 1),
            "flat_threshold": entry.get("flat_threshold"),  # default cut h
            "tmax": entry.get("tmax"),
            "dcoord_min_positive": float(positive.min()) if positive.size else None,
            "dcoord_max": float(dcoord.max()) if dcoord.size else None,
            "n_clusters": int(np.unique(np.asarray(entry["partition"])).size),
        }
    )


def specific_heat_spec(
    result,
    *,
    filter_name: str | None = None,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """Specific heat ``C(τ) = −dS/d·log τ`` from the LRG Laplacian spectrum.

    The spectrum is recomputed online from the (optionally sparsified) filtered
    graph via the same ``prepare → graph → Laplacian`` chain the partitions use,
    so ``C(τ)`` is consistent with the dendrograms and needs no re-ingest. Its
    peak marks the characteristic diffusion scale ``τ*`` (which τ to look at);
    ``τ′ = 1/λ_max`` is the smallest informative scale.
    """
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return {"kind": "specific_heat", "label": result.label, "filter": None, "error": "no LRG filters"}
    if not result.filtered_networks or fname not in result.filtered_networks:
        return {"kind": "specific_heat", "label": result.label, "filter": fname, "error": "no graph for spectrum"}
    graph, _eff = derive_graph(result, fname, sparsify, sparsify_alpha, sparsify_threshold)
    if graph is None or graph.number_of_nodes() < 3:
        return {"kind": "specific_heat", "label": result.label, "filter": fname, "error": "no graph for spectrum"}

    corr = nx.to_numpy_array(graph)
    normalized = getattr(result.config, "normalized_laplacian", True)
    _g, _l, spectrum = prepared_graph_laplacian_spectrum(corr, normalized=normalized)
    if spectrum.size < 3:
        return {"kind": "specific_heat", "label": result.label, "filter": fname, "error": "degenerate spectrum"}

    # ``entropy`` takes log of a diffusion kernel that legitimately underflows to
    # zero at long τ (handled by its nansum); silence the resulting numpy warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        _one_minus_s, c_tau, _var, t = entropy(spectrum, steps=600, t1=-2, t2=5)
    tau = t[:-1]
    tau_prime = tau_star = lambda_max = None
    try:
        scales = characteristic_scales(spectrum, tau, c_tau)
        tau_prime = float(scales["tau_prime"])
        tau_star = float(scales["tau_star"])
        lambda_max = float(scales["lambda_max"])
    except Exception:  # noqa: BLE001 - marks are optional; curve still renders
        pass

    return clean(
        {
            "kind": "specific_heat",
            "label": result.label,
            "filter": fname,
            "tau": tau,
            "C": c_tau,
            "tau_prime": tau_prime,
            "tau_star": tau_star,
            "lambda_max": lambda_max,
        }
    )


def psi_spec(
    result,
    *,
    filter_name: str | None = None,
    tau_index: int = -1,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """Partition-stability index ``Ψ(n)`` at one τ (Villegas dendrogram gap).

    ``Ψ(n)`` measures the log-gap before each branching read from the top; its
    peak is the most stable number of clusters — the optimal cut. Computed live
    from the cached linkage at the selected τ (``compute_optimal_threshold``). The
    x-axis is the number of clusters (branch ``i`` → ``i+2`` clusters);
    ``heights[i]`` is the cut height that yields ``n_clusters[i]`` so the frontend
    can snap the h-slider onto the optimum.
    """
    fname, parts, _nid = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "psi", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    tau_index = max(-len(parts), min(tau_index, len(parts) - 1))
    entry = parts[tau_index]
    linkage = np.asarray(entry["linkage_matrix"], dtype=float)
    if linkage.shape[0] < 3:
        return {"kind": "psi", "label": result.label, "filter": fname, "error": "too few merges for Ψ"}

    try:
        _flat, opt_threshold, psi, opt_branch = compute_optimal_threshold(linkage)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return {"kind": "psi", "label": result.label, "filter": fname, "error": "Ψ undefined for this linkage"}
    psi = np.asarray(psi, dtype=float)
    if psi.size < 1 or not np.all(np.isfinite(psi)):
        return {"kind": "psi", "label": result.label, "filter": fname, "error": "Ψ undefined for this linkage"}

    # Branch i (dendrogram read top-down) -> i+2 clusters; heights aligned so
    # heights[i] is the cut height giving n_clusters[i].
    d_desc = linkage[:, 2][::-1]
    n_clusters = [int(i + 2) for i in range(psi.size)]
    heights = [float(d_desc[i + 1]) for i in range(psi.size)]
    opt_branch = int(opt_branch)

    return clean(
        {
            "kind": "psi",
            "label": result.label,
            "filter": fname,
            "tau": float(entry["tau"]),
            "tau_index": int(tau_index if tau_index >= 0 else len(parts) + tau_index),
            "psi": psi,
            "n_clusters": n_clusters,
            "heights": heights,
            "optimal_threshold": float(opt_threshold),
            "optimal_n_clusters": int(opt_branch + 2),
            "optimal_psi": float(psi[opt_branch]),
        }
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def partition_flow_spec(
    result,
    *,
    filter_name: str | None = None,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """Raster of community membership across τ (rows = nodes, cols = τ).

    Cluster ids are made **cross-τ consistent** (see :func:`_canonical_labels`) so
    a node keeps its colour while it stays in the same evolving community, and rows
    are sorted by their coarsest- then finest-τ community so the raster reads as
    contiguous bands that merge — not the flickering noise raw ids produce.
    """
    fname, parts, node_ids = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "partition_flow", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    partitions = [np.asarray(p["partition"]).astype(int) for p in parts]
    canon, n_comm = _canonical_labels(partitions)  # (n_nodes, n_tau)

    base_names = _leaf_labels(result, canon.shape[0], node_ids)
    # Sort rows by final community, then initial, for contiguous merging bands.
    order = np.lexsort((canon[:, 0], canon[:, -1]))
    z = canon[order]
    names = [base_names[i] for i in order]

    return clean(
        {
            "kind": "partition_flow",
            "label": result.label,
            "filter": fname,
            "z": z,
            "taus": [float(p["tau"]) for p in parts],
            "names": names,
            "n_communities": int(n_comm),
            "n_clusters": [int(np.unique(p).size) for p in partitions],
        }
    )


def sankey_spec(
    result,
    *,
    filter_name: str | None = None,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """Community-flow Sankey across consecutive τ, coloured by canonical community.

    Nodes are laid out in τ columns (left→right) and coloured by their cross-τ
    community; links inherit their source community's colour so a community's
    thread is visually traceable as it merges/splits across scales.
    """
    fname, parts, _nid = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "sankey", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    partitions = [np.asarray(p["partition"]).astype(int) for p in parts]
    taus = [float(p["tau"]) for p in parts]
    canon, _ = _canonical_labels(partitions)
    n_steps = len(partitions)

    def col_maps(step: int):
        """orig cluster id -> (canonical id, size) for one τ column."""
        p, c = partitions[step], canon[:, step]
        cmap: dict[int, int] = {}
        smap: dict[int, int] = {}
        for oc in set(p.tolist()):
            mask = p == oc
            cmap[oc] = int(c[mask][0])
            smap[oc] = int(mask.sum())
        return cmap, smap

    node_labels: list[str] = []
    node_x: list[float] = []
    node_color: list[str] = []
    node_index: dict[tuple[int, int], int] = {}

    def node_id(step: int, oc: int, cmap, smap) -> int:
        key = (step, int(oc))
        if key not in node_index:
            cid = cmap[oc]
            node_index[key] = len(node_labels)
            node_labels.append(f"c{cid}·{smap[oc]}")
            node_x.append(step / max(n_steps - 1, 1))
            node_color.append(_LRG_PALETTE[cid % len(_LRG_PALETTE)])
        return node_index[key]

    src: list[int] = []
    tgt: list[int] = []
    val: list[int] = []
    link_color: list[str] = []
    for step in range(n_steps - 1):
        a, b = partitions[step], partitions[step + 1]
        amap, asize = col_maps(step)
        bmap, bsize = col_maps(step + 1)
        flow: dict[tuple[int, int], int] = {}
        for ca, cb in zip(a.tolist(), b.tolist()):
            flow[(ca, cb)] = flow.get((ca, cb), 0) + 1
        for (ca, cb), count in flow.items():
            src.append(node_id(step, ca, amap, asize))
            tgt.append(node_id(step + 1, cb, bmap, bsize))
            val.append(count)
            link_color.append(_hex_to_rgba(_LRG_PALETTE[amap[ca] % len(_LRG_PALETTE)], 0.35))

    return clean(
        {
            "kind": "sankey",
            "label": result.label,
            "filter": fname,
            "taus": taus,
            "n_steps": n_steps,
            "node_labels": node_labels,
            "node_x": node_x,
            "node_color": node_color,
            "source": src,
            "target": tgt,
            "value": val,
            "link_color": link_color,
        }
    )


def _leaf_meta(result, n_leaves: int, node_ids: list[int] | None) -> list[dict]:
    """Per-leaf ``{short, network, color}`` aligned to the linkage leaf order."""
    nodes = surviving_labels(result)
    fallback = {"short": "node-?", "network": "?", "color": "#888888"}

    def at(i: int) -> dict:
        nd = nodes[i] if 0 <= i < len(nodes) else None
        return {"short": nd["short"], "network": nd["network"], "color": nd["color"]} if nd else fallback

    if node_ids is not None and len(node_ids) == n_leaves:
        return [at(i) for i in node_ids]
    if len(nodes) == n_leaves:
        return [at(i) for i in range(n_leaves)]
    return [{"short": f"node-{i}", "network": "?", "color": "#888888"} for i in range(n_leaves)]


def _leaf_aligned_graph(result, fname, sparsify, alpha, threshold, node_ids):
    """The graph whose node order matches the linkage leaves 0..n-1.

    For the stored-filter hierarchy this is ``prepare → from_numpy_array`` of the
    filtered network (exactly the graph the pipeline built the linkage on); for a
    sparsified backbone it's the re-derived graph (already leaf-ordered).
    """
    normalized = getattr(result.config, "normalized_laplacian", True)
    if sparsify in (None, "", "filter") or node_ids is None:
        stored: nx.Graph = result.filtered_networks[fname]["graph"]
        corr = nx.to_numpy_array(stored)
        g, _L, _spec = prepared_graph_laplacian_spectrum(corr, normalized=normalized)
        return g
    graph, _eff = derive_graph(result, fname, sparsify, alpha, threshold)
    return graph


def lrg_network_spec(
    result,
    *,
    filter_name: str | None = None,
    tau_index: int = -1,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    edge_quantile: float = 0.0,
    **_,
) -> dict:
    """Clustered-network view: LRG-diffusion-distance MDS layout at one τ.

    Node positions come from metric MDS on the LRG diffusion distance at the
    selected τ (so communities sit together — the "LRG-distance-inspired" layout,
    unlike a generic spring layout). Nodes are returned in the SAME order as the
    dendrogram's linkage leaves (0..n-1), so the client colours them by the very
    same cut it applies to the dendrogram — the clustered-network and the
    dendrogram stay in lock-step, and the h-slider recolours both with no refetch.
    Only τ (a new embedding) refetches.
    """
    fname, parts, node_ids = _lrg_entries(
        result, filter_name, sparsify, sparsify_alpha, sparsify_threshold
    )
    if parts is None:
        return {"kind": "lrg_network", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    tau_index = max(-len(parts), min(tau_index, len(parts) - 1))
    entry = parts[tau_index]
    tau = float(entry["tau"])
    linkage = np.asarray(entry["linkage_matrix"], dtype=float)
    n_leaves = int(linkage.shape[0] + 1)

    try:
        graph = _leaf_aligned_graph(
            result, fname, sparsify, sparsify_alpha, sparsify_threshold, node_ids
        )
    except (KeyError, ValueError):
        return {"kind": "lrg_network", "label": result.label, "filter": fname, "error": "no graph for layout"}
    if graph is None or graph.number_of_nodes() != n_leaves:
        return {"kind": "lrg_network", "label": result.label, "filter": fname,
                "error": "graph/linkage leaf mismatch"}

    coords = diffusion_distance_mds(graph, tau)
    node_list = list(graph.nodes())
    strength = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())
    meta = _leaf_meta(result, n_leaves, node_ids)

    nodes_out = []
    for i, nid in enumerate(node_list):
        x, y = (float(coords[i][0]), float(coords[i][1])) if i < len(coords) else (0.0, 0.0)
        m = meta[i]
        nodes_out.append(
            {
                "id": str(i),  # linkage leaf index -> client cut colouring
                "name": m["short"],
                "network": m["network"],
                "color": m["color"],
                "x": x,
                "y": y,
                "degree": int(degree.get(nid, 0)),
                "strength": float(strength.get(nid, 0.0)),
            }
        )

    idx_of = {nid: i for i, nid in enumerate(node_list)}
    all_w = np.array([abs(d.get("weight", 1.0)) for _, _, d in graph.edges(data=True)], dtype=float)
    thresh = float(np.quantile(all_w, edge_quantile)) if (all_w.size and edge_quantile > 0) else 0.0
    edges_out = []
    for u, v, d in graph.edges(data=True):
        w = float(d.get("weight", 1.0))
        if abs(w) >= thresh:
            edges_out.append({"source": str(idx_of[u]), "target": str(idx_of[v]), "weight": w})

    return clean(
        {
            "kind": "lrg_network",
            "label": result.label,
            "filter": fname,
            "sparsify": sparsify or "filter",
            "tau": tau,
            "tau_index": int(tau_index if tau_index >= 0 else len(parts) + tau_index),
            "n_leaves": n_leaves,
            "nodes": nodes_out,
            "edges": edges_out,
            "n_edges_total": int(all_w.size),
            "n_edges_shown": len(edges_out),
            "layout": "lrg",
        }
    )
