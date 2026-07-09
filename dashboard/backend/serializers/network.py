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


def network_spec(
    result, *, filter_name: str | None = None, edge_quantile: float = 0.9, **_
) -> dict:
    """Cytoscape elements for the filtered network: preset layout + thresholded edges."""
    fname = resolve_filter(result, filter_name)
    if fname is None or fname not in result.filtered_networks:
        return {"kind": "network", "label": result.label, "error": "no filtered network"}
    graph: nx.Graph = result.filtered_networks[fname]["graph"]
    nodes_meta = surviving_labels(result)

    # Deterministic layout from the full weighted graph.
    pos = nx.spring_layout(graph, seed=42, weight="weight")

    # Per-node strength + degree for sizing.
    strength = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())

    nodes = []
    for nid in graph.nodes():
        i = int(nid)
        meta = nodes_meta[i] if 0 <= i < len(nodes_meta) else {
            "short": f"node-{i}", "network": "?", "color": "#888888"
        }
        x, y = pos[nid]
        nodes.append(
            {
                "id": str(i),
                "name": meta["short"],
                "network": meta["network"],
                "color": meta["color"],
                "x": float(x) * 600.0,
                "y": float(y) * 600.0,
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
            "nodes": nodes,
            "edges": edges,
            "n_edges_total": int(all_w.size),
            "n_edges_shown": len(edges),
            "edge_quantile": edge_quantile,
        }
    )
