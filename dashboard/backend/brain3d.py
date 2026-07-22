"""Interactive 3-D brain views (nilearn) for a filtered network.

Renders the per-filter network as a rotatable 3-D figure: ``connectome`` (edges
from the thresholded adjacency drawn between survivor parcel centroids) or
``markers`` (parcel nodes coloured by their 7-network). Output is a self-contained
nilearn HTML page served straight into an ``<iframe>`` (so the 2.5 MB payload
never round-trips as JSON). Node identity flows through
:mod:`dashboard.backend.remap` exactly as the 2-D network/heatmap specs do — graph
node IDs index the survivor-aligned coords/labels.

nilearn is already a core dependency; it is imported lazily so non-3-D paths
never pay for it.
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx

from . import remap
from .catalog import dataset_dir
from .loaders import get_results
from .remap import surviving_labels
from .serializers.network import resolve_filter


def _error_page(message: str) -> str:
    return (
        "<!doctype html><html><body style='font-family:system-ui;color:#c62828;"
        "padding:16px'>" + message + "</body></html>"
    )


def _build_html(
    result, fname: str, mode: str, edge_quantile: float, node_size: float
) -> str:
    from nilearn import plotting

    graph: nx.Graph = result.filtered_networks[fname]["graph"]
    ids = sorted(int(x) for x in graph.nodes())
    if not ids:
        return _error_page("This network has no nodes to display.")

    coords = remap.surviving_coords(result)
    if coords.shape[0] <= max(ids):
        return _error_page("Atlas centroids are unavailable for this result.")
    node_coords = coords[ids]

    labels = surviving_labels(result)
    colors = [labels[i]["color"] if 0 <= i < len(labels) else "#888888" for i in ids]
    names = [labels[i]["short"] if 0 <= i < len(labels) else f"node-{i}" for i in ids]

    if mode == "markers":
        # Markers render a touch larger than connectome nodes at the same setting.
        view = plotting.view_markers(
            node_coords, marker_color=colors, marker_labels=names,
            marker_size=node_size * 1.3,
        )
    else:  # connectome
        adjacency = nx.to_numpy_array(graph, nodelist=ids, weight="weight")
        pct = max(0.0, min(99.9, float(edge_quantile) * 100.0))
        view = plotting.view_connectome(
            adjacency,
            node_coords,
            edge_threshold=f"{pct:.1f}%",
            node_color=colors,
            node_size=node_size,
            symmetric_cmap=False,
        )
    return view.get_standalone()


@lru_cache(maxsize=64)
def _html_cached(
    dataset: str, label: str, fname: str, mode: str, edge_quantile: float,
    node_size: float, _mtime: float,
) -> str:
    rc = get_results(dataset_dir(dataset))
    return _build_html(rc[label], fname, mode, edge_quantile, node_size)


def render(
    dataset: str,
    label: str,
    filter_name: str | None,
    mode: str = "connectome",
    edge_quantile: float = 0.98,
    node_size: float = 9.0,
) -> str:
    """Standalone HTML for the 3-D brain, or a small HTML error page."""
    directory = dataset_dir(dataset)
    if directory is None:
        return _error_page(f"Unknown dataset {dataset!r}.")
    rc = get_results(directory)
    try:
        result = rc[label]
    except KeyError:
        return _error_page(f"Unknown result {label!r}.")
    fname = resolve_filter(result, filter_name)
    if fname is None or fname not in result.filtered_networks:
        return _error_page("No filtered network available for this result.")
    mode = "markers" if mode == "markers" else "connectome"
    mtime = (directory / "results.pkl").stat().st_mtime
    return _html_cached(
        dataset, label, fname, mode, round(float(edge_quantile), 3),
        round(float(node_size), 1), mtime,
    )
