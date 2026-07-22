"""Serializers for Section-3 (standard network) plots, per filter method.

Filter selection is tolerant: an unknown/absent ``filter_name`` falls back to the
first available filter. The 2-D network graph is laid out server-side
(deterministic spring layout) and edge-thresholded so dense backbones stay
legible; the frontend (Cytoscape) renders the preset positions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..encode import clean
from ..remap import surviving_labels


def resolve_filter(result, filter_name: str | None) -> str | None:
    """Pick *filter_name* if present, else the first available network filter."""
    filters = list(result.network_analyses.keys())
    if not filters:
        return None
    if filter_name in filters:
        return filter_name
    return filters[0]


def global_metrics_spec(result, *, filter_name: str | None = None, **_) -> dict:
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return {"kind": "global_metrics", "label": result.label, "error": "no network analyses"}
    gm = result.network_analyses[fname].get("global_metrics", {})
    return clean(
        {"kind": "global_metrics", "label": result.label, "filter": fname, "metrics": gm}
    )


def degree_distribution_spec(result, *, filter_name: str | None = None, **_) -> dict:
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return {"kind": "degree_distribution", "label": result.label, "error": "no network analyses"}
    dd = result.network_analyses[fname].get("degree_distribution", {})
    hist = dd.get("histogram")
    counts, edges = (hist[0], hist[1]) if hist is not None else (None, None)
    return clean(
        {
            "kind": "degree_distribution",
            "label": result.label,
            "filter": fname,
            "counts": counts,
            "edges": edges,
            "values": dd.get("values"),
            "mean": dd.get("mean"),
            "std": dd.get("std"),
            "min": dd.get("min"),
            "max": dd.get("max"),
        }
    )


def node_metrics_spec(result, *, filter_name: str | None = None, **_) -> dict:
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return {"kind": "node_metrics", "label": result.label, "error": "no network analyses"}
    df = result.network_analyses[fname].get("node_metrics")
    if df is None:
        return {"kind": "node_metrics", "label": result.label, "filter": fname, "error": "no node metrics"}
    nodes = surviving_labels(result)
    names = []
    for idx in df.index:
        i = int(idx)
        names.append(nodes[i]["short"] if 0 <= i < len(nodes) else f"node-{i}")
    return clean(
        {
            "kind": "node_metrics",
            "label": result.label,
            "filter": fname,
            "columns": list(df.columns),
            "names": names,
            "rows": df.to_numpy(),
        }
    )


def _derive_graph(
    result, fname: str, sparsify: str | None, alpha: float, threshold: float
) -> tuple[nx.Graph, str]:
    """Return the graph to display + the effective sparsification label.

    ``sparsify`` selects a backbone re-derived from the signed correlation
    (``corr_prepared``) with the pipeline's own methods so dense hairballs can be
    thinned to a legible skeleton; ``None``/``"filter"`` keeps the pre-computed
    filtered network unchanged. Falls back to the filtered network if the signed
    matrix isn't stored (older bundles) or the backbone collapses.
    """
    stored: nx.Graph = result.filtered_networks[fname]["graph"]
    if sparsify in (None, "", "filter"):
        return stored, "filter"
    corr = getattr(result, "corr_prepared", None)
    if corr is None:
        return stored, "filter"
    corr = np.asarray(corr, dtype=float)
    try:
        if sparsify == "percolation":
            from multifunbrain.processing.backbone import filter_validated

            g, _ = filter_validated(corr, method="percolation", weights="abs")
        elif sparsify == "disparity":
            from multifunbrain.processing.backbone import filter_validated

            g, _ = filter_validated(corr, method="disparity", alpha=alpha, weights="positive")
        elif sparsify == "threshold":
            from multifunbrain.processing.filtering import filter_absolute_threshold

            g, _ = filter_absolute_threshold(corr, threshold=threshold)
        else:
            return stored, "filter"
    except (ValueError, np.linalg.LinAlgError):
        return stored, "filter"
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return stored, "filter"
    return g, sparsify


def _layouts(graph: nx.Graph, node_ids: list, scale: float = 600.0) -> dict:
    """A few named 2-D layouts (positions aligned to *node_ids*) for the client.

    Force-directed reveals modular structure; Kamada-Kawai often untangles small
    dense graphs; spectral places nodes by the graph Laplacian's low modes. Any
    that fail (e.g. Kamada-Kawai on a pathological graph) are simply omitted.
    """
    out: dict[str, list] = {}

    def collect(fn):
        try:
            pos = fn()
        except Exception:  # noqa: BLE001 - a failed layout is just dropped
            return None
        return [[float(pos[n][0]) * scale, float(pos[n][1]) * scale] for n in node_ids]

    out["spring"] = collect(lambda: nx.spring_layout(graph, seed=42, weight="weight"))
    if len(node_ids) <= 600:
        out["kamada"] = collect(lambda: nx.kamada_kawai_layout(graph, weight="weight"))
    out["spectral"] = collect(lambda: nx.spectral_layout(graph, weight="weight"))
    return {k: v for k, v in out.items() if v}


def network_spec(
    result,
    *,
    filter_name: str | None = None,
    edge_quantile: float = 0.9,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    **_,
) -> dict:
    """Cytoscape elements for the network: server layouts + optional sparsification.

    ``sparsify`` (``percolation`` | ``disparity`` | ``threshold``) re-derives a
    sparse backbone from the signed correlation; ``edge_quantile`` then thins the
    drawn edges by |weight| so even the stored dense network stays readable.
    """
    fname = resolve_filter(result, filter_name)
    if fname is None or fname not in result.filtered_networks:
        return {"kind": "network", "label": result.label, "error": "no filtered network"}
    graph, eff_sparsify = _derive_graph(
        result, fname, sparsify, sparsify_alpha, sparsify_threshold
    )
    nodes_meta = surviving_labels(result)

    node_ids = list(graph.nodes())
    layouts = _layouts(graph, node_ids)
    spring = layouts.get("spring")

    # Per-node strength + degree for sizing / colouring.
    strength = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())

    nodes = []
    for i, nid in enumerate(node_ids):
        idx = int(nid)
        meta = nodes_meta[idx] if 0 <= idx < len(nodes_meta) else {
            "short": f"node-{idx}", "network": "?", "color": "#888888"
        }
        x, y = (spring[i] if spring else (0.0, 0.0))
        nodes.append(
            {
                "id": str(idx),
                "name": meta["short"],
                "network": meta["network"],
                "color": meta["color"],
                "x": float(x),
                "y": float(y),
                "degree": int(degree.get(nid, 0)),
                "strength": float(strength.get(nid, 0.0)),
            }
        )

    # Threshold edges by |weight| quantile so dense backbones stay readable.
    all_w = np.array([abs(d.get("weight", 1.0)) for _, _, d in graph.edges(data=True)])
    thresh = float(np.quantile(all_w, edge_quantile)) if all_w.size else 0.0
    edges = []
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1.0)
        if abs(w) >= thresh:
            edges.append({"source": str(int(u)), "target": str(int(v)), "weight": float(w)})

    return clean(
        {
            "kind": "network",
            "label": result.label,
            "filter": fname,
            "sparsify": eff_sparsify,
            "nodes": nodes,
            "edges": edges,
            "layouts": layouts,
            "n_nodes": len(nodes),
            "n_edges_total": int(all_w.size),
            "n_edges_shown": len(edges),
            "edge_quantile": edge_quantile,
        }
    )
