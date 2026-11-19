"""Section 3 plots — standard unsigned-network visualisations.

Per-node metric distributions, signed-network layout, and filtered
network drawings (community-coloured nodes, edge widths scaled by
weight).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._helpers import _network_layout, _resolve_filter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_network",
    "plot_node_metrics",
    "plot_signed_network",
]


def plot_node_metrics(
    result: PipelineResult,
    filter_name: str | None = None,
    *,
    metrics: tuple[str, ...] = ("degree", "strength", "clustering", "betweenness"),
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Distribution plots for per-node metrics of a filtered network.

    Parameters
    ----------
    result : PipelineResult
        Must have ``network_analyses`` populated.
    filter_name : str or None
        Which filtered network to plot. Defaults to the first available.
    metrics : tuple of str
        Node metric columns to plot.

    Returns
    -------
    (fig, axes)
        Array of axes, one per metric.
    """
    if filter_name is None:
        filter_name = next(iter(result.network_analyses))
    analysis = result.network_analyses[filter_name]
    node_df = analysis.get("node_metrics")
    if node_df is None:
        raise ValueError(f"No node_metrics found for filter '{filter_name}'")

    available = [m for m in metrics if m in node_df.columns]
    n = len(available)
    if figsize is None:
        figsize = (4 * n, 3.5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = np.array([axes])

    for ax, metric in zip(axes, available):
        vals = node_df[metric].dropna().values
        ax.hist(vals, bins=20, color="#5C6BC0", edgecolor="k", linewidth=0.5, alpha=0.8)
        ax.axvline(np.mean(vals), color="#F44336", linestyle="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Node metrics: {filter_name}", fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_signed_network(
    result: PipelineResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw the raw signed correlation network.

    Positive edges in blue, negative in red. Edge width/alpha scale
    with |weight|. Nodes are colored by the Louvain partition of the
    absolute-value network when available.

    Returns
    -------
    (fig, ax)
    """
    corr = result.corr_prepared
    G = nx.Graph()
    n = corr.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = corr[i, j]
            if w != 0.0:
                G.add_edge(i, j, weight=w)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos = _network_layout(G, signed=True)

    node_colors = "#78909C"
    if result.network_analyses:
        first = next(iter(result.network_analyses))
        comm = result.network_analyses[first].get("community", {})
        partition = comm.get("partition")
        if partition:
            cmap = plt.cm.tab20
            node_colors = [cmap(partition.get(i, 0) % 20) for i in range(n)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=node_colors,
                           edgecolors="k", linewidths=0.3)

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    wmax = max(abs(d["weight"]) for _, _, d in G.edges(data=True)) if G.edges else 1.0

    if pos_edges:
        pw = [abs(G[u][v]["weight"]) / wmax * 1.5 for u, v in pos_edges]
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                               width=pw, edge_color="#1565C0", alpha=0.25)
    if neg_edges:
        nw = [abs(G[u][v]["weight"]) / wmax * 1.5 for u, v in neg_edges]
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                               width=nw, edge_color="#C62828", alpha=0.25)

    ax.set_title(f"Signed network ({n} nodes, {G.number_of_edges()} edges)")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_network(
    result: PipelineResult,
    filter_name: str | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw a filtered network with community-colored nodes.

    Nodes colored by Louvain partition, sized by strength.
    Edge widths scale with weight.

    Returns
    -------
    (fig, ax)
    """
    fname = _resolve_filter(result, filter_name)
    G = result.filtered_networks[fname]["graph"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos = _network_layout(G)
    nodes = list(G.nodes())

    analysis = result.network_analyses.get(fname, {})
    partition = analysis.get("community", {}).get("partition", {})
    cmap = plt.cm.tab20
    node_colors = [cmap(partition.get(n, 0) % 20) for n in nodes]

    node_df = analysis.get("node_metrics")
    if node_df is not None and "strength" in node_df.columns:
        strengths = node_df["strength"].reindex(nodes).fillna(1.0).values
        smin, smax = strengths.min(), strengths.max()
        if smax > smin:
            sizes = 20 + 200 * (strengths - smin) / (smax - smin)
        else:
            sizes = 60
    else:
        sizes = 60

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors,
                           edgecolors="k", linewidths=0.3)

    wmax = max(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) if G.edges else 1.0
    widths = [d.get("weight", 0.5) / wmax * 1.5 for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#90A4AE", alpha=0.3)

    n_comm = len(set(partition.values())) if partition else "?"
    ax.set_title(f"{fname} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                 f"{n_comm} communities)")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax
