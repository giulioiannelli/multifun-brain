"""Serializers for the LRG multiscale section (per filter) of a PipelineResult.

A result's ``lrg_results[filter]`` is a list of per-tau dicts
(``tau``, ``partition``, ``linkage_matrix``, ``flat_threshold``, ``tmax``) or an
error dict. From these we derive an interactive dendrogram (SciPy layout),
a partition-flow raster across tau, and a community-flow Sankey. (Specific-heat /
entropy / PSI curves come from standalone MultiscaleResult bundles, added later.)
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import dendrogram

from ..encode import clean
from ..remap import surviving_labels


def resolve_filter(result, filter_name: str | None) -> str | None:
    filters = list(result.lrg_results.keys())
    if not filters:
        return None
    if filter_name in filters:
        return filter_name
    return filters[0]


def _entries(result, filter_name):
    """Return (filter, list-of-tau-entries) or (filter, None) on error/empty."""
    fname = resolve_filter(result, filter_name)
    if fname is None:
        return None, None
    parts = result.lrg_results[fname]
    if isinstance(parts, dict):  # error sentinel
        return fname, None
    if not isinstance(parts, list) or not parts:
        return fname, None
    return fname, parts


def tau_grid_spec(result, *, filter_name: str | None = None, **_) -> dict:
    """List of available tau steps (for the LRG tau selector)."""
    fname, parts = _entries(result, filter_name)
    if parts is None:
        return {"kind": "tau_grid", "label": result.label, "filter": fname, "taus": []}
    taus = [float(p["tau"]) for p in parts]
    nclusters = [int(np.unique(np.asarray(p["partition"])).size) for p in parts]
    return clean(
        {"kind": "tau_grid", "label": result.label, "filter": fname,
         "taus": taus, "n_clusters": nclusters}
    )


def dendrogram_spec(result, *, filter_name: str | None = None, tau_index: int = -1, **_) -> dict:
    fname, parts = _entries(result, filter_name)
    if parts is None:
        return {"kind": "dendrogram", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    tau_index = max(-len(parts), min(tau_index, len(parts) - 1))
    entry = parts[tau_index]
    linkage = np.asarray(entry["linkage_matrix"])
    nodes = surviving_labels(result)
    labels = [nd["short"] for nd in nodes]
    if len(labels) != linkage.shape[0] + 1:
        labels = [f"node-{i}" for i in range(linkage.shape[0] + 1)]

    dn = dendrogram(linkage, no_plot=True, labels=labels)
    dcoord = np.asarray(dn["dcoord"], dtype=float)
    positive = dcoord[dcoord > 0]
    return clean(
        {
            "kind": "dendrogram",
            "label": result.label,
            "filter": fname,
            "tau": float(entry["tau"]),
            "tau_index": int(tau_index if tau_index >= 0 else len(parts) + tau_index),
            "icoord": dn["icoord"],
            "dcoord": dn["dcoord"],
            "ivl": dn["ivl"],
            "color_list": dn["color_list"],
            "flat_threshold": entry.get("flat_threshold"),
            "tmax": entry.get("tmax"),
            "dcoord_min_positive": float(positive.min()) if positive.size else None,
            "n_clusters": int(np.unique(np.asarray(entry["partition"])).size),
        }
    )


def partition_flow_spec(result, *, filter_name: str | None = None, **_) -> dict:
    """Raster: rows = nodes, cols = tau, value = cluster id (interactive heatmap)."""
    fname, parts = _entries(result, filter_name)
    if parts is None:
        return {"kind": "partition_flow", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    z = np.column_stack([np.asarray(p["partition"]) for p in parts])  # (n_nodes, n_tau)
    nodes = surviving_labels(result)
    names = [nd["short"] for nd in nodes] if len(nodes) == z.shape[0] else [f"node-{i}" for i in range(z.shape[0])]
    return clean(
        {
            "kind": "partition_flow",
            "label": result.label,
            "filter": fname,
            "z": z,
            "taus": [float(p["tau"]) for p in parts],
            "names": names,
            "n_clusters": [int(np.unique(np.asarray(p["partition"])).size) for p in parts],
        }
    )


def sankey_spec(result, *, filter_name: str | None = None, **_) -> dict:
    """Community-flow Sankey across consecutive tau steps."""
    fname, parts = _entries(result, filter_name)
    if parts is None:
        return {"kind": "sankey", "label": result.label, "filter": fname, "error": "no LRG partitions"}
    partitions = [np.asarray(p["partition"]) for p in parts]
    taus = [float(p["tau"]) for p in parts]

    node_labels: list[str] = []
    node_index: dict[tuple[int, int], int] = {}

    def node_id(step: int, cluster: int) -> int:
        key = (step, int(cluster))
        if key not in node_index:
            node_index[key] = len(node_labels)
            node_labels.append(f"τ{step}:c{int(cluster)}")
        return node_index[key]

    src: list[int] = []
    tgt: list[int] = []
    val: list[int] = []
    for step in range(len(partitions) - 1):
        a, b = partitions[step], partitions[step + 1]
        flow: dict[tuple[int, int], int] = {}
        for ca, cb in zip(a.tolist(), b.tolist()):
            flow[(ca, cb)] = flow.get((ca, cb), 0) + 1
        for (ca, cb), count in flow.items():
            src.append(node_id(step, ca))
            tgt.append(node_id(step + 1, cb))
            val.append(count)

    return clean(
        {
            "kind": "sankey",
            "label": result.label,
            "filter": fname,
            "taus": taus,
            "node_labels": node_labels,
            "source": src,
            "target": tgt,
            "value": val,
        }
    )
