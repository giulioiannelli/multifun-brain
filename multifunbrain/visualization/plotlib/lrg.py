"""LRG multiscale plots.

Combines the per-result LRG plots (operate on
:class:`~multifunbrain.pipeline.PipelineResult`) and the multiscale
plots (operate on
:class:`~multifunbrain.pipeline.multiscale.MultiscaleResult`).

Per-result plots (Section 3 of the per-matrix pipeline):
    plot_lrg_entropy, plot_lrg_dendrogram, plot_lrg_psi,
    plot_lrg_partition_network, plot_lrg_sankey.

Multiscale plots (one or two MultiscaleResults):
    plot_specific_heat, plot_specific_heat_overlay,
    plot_dendrogram_with_psi, plot_psi_curve, plot_rmi_curve,
    plot_partition_flow, plot_tanglegram, plot_metastable_overlay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from ...analysis.lrg.compare import (
    reduced_mutual_information,
    specific_heat_comparison,
)
from ...pipeline.multiscale import MultiscaleResult
from ..style import FIGSIZE, PALETTES
from ._helpers import _nearest_tau_index, _network_layout, _resolve_filter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_dendrogram_with_psi",
    "plot_lrg_dendrogram",
    "plot_lrg_entropy",
    "plot_lrg_partition_network",
    "plot_lrg_psi",
    "plot_lrg_sankey",
    "plot_metastable_overlay",
    "plot_partition_flow",
    "plot_psi_curve",
    "plot_rmi_curve",
    "plot_specific_heat",
    "plot_specific_heat_overlay",
    "plot_tanglegram",
]


# ──────────────────────────────────────────────────────────────────────
# Per-result LRG plots (one PipelineResult)
# ──────────────────────────────────────────────────────────────────────


def plot_lrg_entropy(
    result: PipelineResult,
    filter_name: str | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Entropy (1-S) and specific heat (C) on dual y-axes.

    Recomputes the Laplacian spectrum from the stored filtered graph,
    then calls :func:`multifunbrain.analysis.lrg.kernel.entropy`.

    Returns
    -------
    (fig, ax)
    """
    from ...analysis.lrg.kernel import entropy as lrg_entropy
    from ...analysis.lrg.kernel import graph_laplacian_and_spectrum
    from .entropy import plot_entropy_and_C

    fname = _resolve_filter(result, filter_name)
    G = result.filtered_networks[fname]["graph"]
    _, spectrum = graph_laplacian_and_spectrum(G, normalized=True)

    Sm1, dS, _varL, t = lrg_entropy(spectrum)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    plot_entropy_and_C(ax, t, Sm1, dS)
    ax.set_title(f"LRG entropy & specific heat ({fname})")
    fig.tight_layout()
    return fig, ax


def plot_lrg_dendrogram(
    result: PipelineResult,
    filter_name: str | None = None,
    tau_index: int = -1,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Dendrogram from LRG hierarchical clustering.

    Draws the dendrogram for the partition at ``tau_index`` and a
    horizontal line at the flat clustering threshold.

    Returns
    -------
    (fig, ax)
    """
    fname = _resolve_filter(result, filter_name)
    parts = result.lrg_results.get(fname)
    if not parts:
        raise ValueError(f"No LRG results for filter '{fname}'.")

    entry = parts[tau_index]
    Z = entry["linkage_matrix"]
    threshold = entry["flat_threshold"]
    tau = entry["tau"]

    if ax is None:
        # Template: DENDROGRAM uses a wide figure so leaf labels fit.
        fig, ax = plt.subplots(figsize=(12, FIGSIZE["wide"][1] + 1))
    else:
        fig = ax.get_figure()

    dendrogram(Z, ax=ax, color_threshold=threshold, leaf_font_size=6)
    # The cut indicator uses the Material "danger" colour from the
    # central palette so every dendrogram in the project is visually
    # consistent.
    ax.axhline(threshold,
               color=PALETTES["material"]["danger_light"],
               linestyle="--", linewidth=1.2,
               label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Node")
    ax.set_ylabel("Distance")
    n_clusters = len(np.unique(entry["partition"]))
    ax.set_title(f"LRG dendrogram ({fname}, tau={tau:.3g}, {n_clusters} clusters)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_lrg_psi(
    result: PipelineResult,
    filter_name: str | None = None,
    tau_index: int = -1,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Partition Stability Index (PSI) vs branch index.

    Recomputes PSI from the stored linkage matrix and marks the
    optimal branch.

    Returns
    -------
    (fig, ax)
    """
    from ...analysis.lrg.partitions import compute_optimal_threshold

    fname = _resolve_filter(result, filter_name)
    parts = result.lrg_results.get(fname)
    if not parts:
        raise ValueError(f"No LRG results for filter '{fname}'.")

    entry = parts[tau_index]
    Z = entry["linkage_matrix"]
    tau = entry["tau"]

    _, _, stability_indices, optimal_idx = compute_optimal_threshold(Z)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.bar(np.arange(len(stability_indices)), stability_indices,
           color="#5C6BC0", edgecolor="k", linewidth=0.3)
    ax.axvline(optimal_idx, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"optimal branch={optimal_idx}")
    ax.set_xlabel("Branch index")
    ax.set_ylabel("Stability index")
    ax.set_title(f"Partition Stability Index ({fname}, tau={tau:.3g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_lrg_partition_network(
    result: PipelineResult,
    filter_name: str | None = None,
    tau_index: int = -1,
    *,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Network with nodes colored by LRG partition.

    Uses the partition at ``tau_index`` from the LRG results for the
    given filter.

    Returns
    -------
    (fig, ax)
    """
    fname = _resolve_filter(result, filter_name)
    parts = result.lrg_results.get(fname)
    if not parts:
        raise ValueError(f"No LRG results for filter '{fname}'.")

    entry = parts[tau_index]
    partition = entry["partition"]
    tau = entry["tau"]
    G = result.filtered_networks[fname]["graph"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos = _network_layout(G)
    nodes = list(G.nodes())

    cmap = plt.cm.tab20
    node_colors = [cmap(int(partition[i]) % 20) for i, _ in enumerate(nodes)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=node_colors,
                           edgecolors="k", linewidths=0.3)
    wmax = max(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) if G.edges else 1.0
    widths = [d.get("weight", 0.5) / wmax * 1.5 for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#90A4AE", alpha=0.3)

    n_clusters = len(np.unique(partition))
    ax.set_title(f"LRG partition ({fname}, tau={tau:.3g}, {n_clusters} clusters)")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_lrg_sankey(
    result: PipelineResult,
    filter_name: str | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Sankey diagram of community evolution across tau scales.

    Reuses the matplotlib Sankey implementation in
    :mod:`multifunbrain.visualization.plotlib.sankey`.

    Returns
    -------
    (fig, ax)
    """
    from .sankey import plot_sankey

    fname = _resolve_filter(result, filter_name)
    parts = result.lrg_results.get(fname)
    if not parts:
        raise ValueError(f"No LRG results for filter '{fname}'.")

    partitions = [p["partition"] for p in parts]
    tau_values = [p["tau"] for p in parts]

    plot_sankey(partitions, tau_values, backend="matplotlib")
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None
    fig.suptitle(f"LRG Sankey ({fname})", fontsize=12)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Multiscale plots (one or two MultiscaleResults)
# ──────────────────────────────────────────────────────────────────────


def plot_specific_heat(
    result: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    mark_tau_prime: bool = True,
    mark_tau_star: bool = True,
    title: str | None = None,
    color: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """``C(τ)`` curve with τ′ and τ* marked.

    Template: SPECIFIC_HEAT pulls all colours from ``style.PALETTES``
    so the trace, τ′ line, and τ* line are visually consistent across
    every figure in the project. Pass ``color=...`` to override the
    trace colour for an overlay (e.g. when stacking two curves).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    else:
        fig = ax.get_figure()

    trace_color = color if color is not None else PALETTES["material"]["primary_dark"]
    ax.semilogx(result.time_grid_broad, result.specific_heat, "-",
                color=trace_color, linewidth=1.4)
    ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)

    if mark_tau_prime and np.isfinite(result.tau_prime):
        ax.axvline(result.tau_prime,
                   color=PALETTES["material"]["accent"],
                   linestyle="--", linewidth=1.0, alpha=0.8,
                   label=r"$\tau'=1/\lambda_{max}$")
    if mark_tau_star and np.isfinite(result.tau_star):
        ax.axvline(result.tau_star,
                   color=PALETTES["material"]["success"],
                   linestyle=":", linewidth=1.2, alpha=0.9,
                   label=r"$\tau^*$")

    fs = 8 if compact else 10
    ax.set_xlabel(r"$\tau$", fontsize=fs)
    ax.set_ylabel(r"$C(\tau)$", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=fs)
    if mark_tau_prime or mark_tau_star:
        ax.legend(fontsize=fs - 2, loc="best")
    return fig, ax


def plot_specific_heat_overlay(
    result_a: MultiscaleResult,
    result_b: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    labels: tuple[str, str] | None = None,
    colors: tuple[str, str] = ("#C62828", "#1565C0"),
    annotate: bool = True,
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Overlay ``C_a(τ)`` and ``C_b(τ)`` with τ* shift annotation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
    else:
        fig = ax.get_figure()

    if labels is None:
        labels = (result_a.label or "A", result_b.label or "B")

    ax.semilogx(result_a.time_grid_broad, result_a.specific_heat, "-",
                color=colors[0], linewidth=1.4, label=labels[0])
    ax.semilogx(result_b.time_grid_broad, result_b.specific_heat, "-",
                color=colors[1], linewidth=1.4, label=labels[1])
    ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(result_a.tau_star, color=colors[0], linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axvline(result_b.tau_star, color=colors[1], linestyle=":", linewidth=0.9, alpha=0.7)

    fs = 8 if compact else 10
    if annotate:
        cmp = specific_heat_comparison(
            result_a.specific_heat, result_b.specific_heat, result_a.time_grid_broad
        )
        ax.text(
            0.02, 0.98,
            rf"$\Delta\log\tau^*={cmp['log_delta_tau_star']:+.2f}$"
            "\n"
            rf"area$|\Delta C|={cmp['area_abs_delta']:.2f}$",
            transform=ax.transAxes, fontsize=fs - 1, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.set_xlabel(r"$\tau$", fontsize=fs)
    ax.set_ylabel(r"$C(\tau)$", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs - 1)
    ax.legend(fontsize=fs - 2, loc="upper right")
    if title:
        ax.set_title(title, fontsize=fs)
    return fig, ax


def plot_dendrogram_with_psi(
    result: MultiscaleResult,
    *,
    tau_value: float | None = None,
    ax_top: matplotlib.axes.Axes | None = None,
    ax_bot: matplotlib.axes.Axes | None = None,
    color_threshold: float | None = None,
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Paper Fig 2 (a + c) layout: dendrogram on top, ``Ψ(n; τ)`` below.

    If ``tau_value`` is ``None`` use ``τ′ = 1/λ_max``.
    """
    if not result.per_tau:
        raise ValueError(f"MultiscaleResult {result.label!r} has no per-τ entries.")
    if tau_value is None:
        tau_value = result.tau_prime
    idx = _nearest_tau_index(result.tau_grid, tau_value)
    entry = result.per_tau[idx]

    if ax_top is None and ax_bot is None:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(5.0, 4.5),
            gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.05},
            sharex=False,
        )
    elif ax_top is None or ax_bot is None:
        raise ValueError("Provide both ax_top and ax_bot, or neither.")
    else:
        fig = ax_top.get_figure()

    thresh = color_threshold if color_threshold is not None else entry["flat_threshold"]
    dendrogram(
        entry["linkage"],
        ax=ax_top,
        color_threshold=thresh,
        above_threshold_color="grey",
        no_labels=True,
    )
    ax_top.axhline(thresh, color="#C62828", linewidth=1.0, linestyle="--", alpha=0.8)
    fs = 8 if compact else 10
    ax_top.set_ylabel(r"$\Delta$", fontsize=fs)
    ax_top.set_yscale("log")
    ax_top.tick_params(axis="both", labelsize=fs - 1)
    ax_top.set_xticks([])
    title_str = (
        title if title is not None
        else f"{result.label or ''}  τ={entry['tau']:.2g}  n*={entry['n_clusters']}"
    )
    ax_top.set_title(title_str, fontsize=fs)

    psi = entry["psi_n"]
    n_axis = np.arange(2, len(psi) + 2)
    ax_bot.plot(n_axis, psi, "o-", color="#2E7D32", markersize=3, linewidth=1.0)
    if entry["optimal_branch_idx"] is not None:
        n_opt = entry["optimal_branch_idx"] + 2
        ax_bot.axvline(n_opt, color="#C62828", linestyle="--", alpha=0.7, linewidth=0.8)
    ax_bot.set_xlabel("n (gap index)", fontsize=fs)
    ax_bot.set_ylabel(r"$\Psi(n;\tau)$", fontsize=fs)
    ax_bot.tick_params(axis="both", labelsize=fs - 1)
    ax_bot.set_xlim(1, min(20, len(psi) + 2))
    return fig, (ax_top, ax_bot)


def plot_psi_curve(
    result: MultiscaleResult,
    *,
    tau_value: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """``Ψ(n; τ)`` alone, for a chosen τ. Paper Fig 2(c)."""
    if not result.per_tau:
        raise ValueError(f"MultiscaleResult {result.label!r} has no per-τ entries.")
    if tau_value is None:
        tau_value = result.tau_prime
    idx = _nearest_tau_index(result.tau_grid, tau_value)
    entry = result.per_tau[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
    else:
        fig = ax.get_figure()

    psi = entry["psi_n"]
    n_axis = np.arange(2, len(psi) + 2)
    ax.plot(n_axis, psi, "o-", color="#2E7D32", markersize=3, linewidth=1.0)
    n_opt = entry["optimal_branch_idx"] + 2
    ax.axvline(n_opt, color="#C62828", linestyle="--", alpha=0.7, linewidth=0.8,
               label=f"n*={entry['n_clusters']}")
    fs = 8 if compact else 10
    ax.set_xlabel("n", fontsize=fs)
    ax.set_ylabel(r"$\Psi(n;\tau)$", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs - 1)
    ax.set_xlim(1, min(20, len(psi) + 2))
    ax.legend(fontsize=fs - 2)
    if title:
        ax.set_title(title, fontsize=fs)
    return fig, ax


def plot_rmi_curve(
    result_a: MultiscaleResult,
    result_b: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    labels: tuple[str, str] | None = None,
    color: str = "#6A1B9A",
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Reduced Mutual Information between optimal partitions at matched τ.

    Assumes ``result_a`` and ``result_b`` share the same ``tau_grid``
    (length) — within the April pipeline they're produced by the same
    runner so this holds by construction.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
    else:
        fig = ax.get_figure()

    tau_a = result_a.tau_grid
    tau_b = result_b.tau_grid
    n = min(len(tau_a), len(tau_b))
    rmi_vals = np.empty(n)
    for k in range(n):
        rmi_vals[k] = reduced_mutual_information(
            result_a.per_tau[k]["partition"],
            result_b.per_tau[k]["partition"],
        )

    ax.semilogx(tau_a[:n], rmi_vals, "o-", color=color, markersize=3, linewidth=1.2)
    ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)

    fs = 8 if compact else 10
    ax.set_xlabel(r"$\tau$", fontsize=fs)
    ax.set_ylabel("RMI (nats)", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=fs)
    if labels is not None:
        ax.text(0.02, 0.98, f"{labels[0]} vs {labels[1]}",
                transform=ax.transAxes, fontsize=fs - 2, va="top", ha="left",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
    return fig, ax


def plot_partition_flow(
    result: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap_name: str = "tab20",
    highlight_metastable: bool = True,
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Per-node × per-τ partition raster — visual proxy for a Sankey.

    Each row is a node, each column a τ, and the cell colour encodes
    the cluster label at that τ. Metastable nodes (those whose
    membership-set changes) are marked with a vertical red tick at the
    row's far right.
    """
    if not result.per_tau:
        raise ValueError(f"MultiscaleResult {result.label!r} has no per-τ entries.")

    n_nodes = result.per_tau[0]["partition"].size
    n_tau = len(result.per_tau)
    raster = np.zeros((n_nodes, n_tau), dtype=int)
    for k, entry in enumerate(result.per_tau):
        raster[:, k] = entry["partition"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 0.04 * n_nodes + 1.6))
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap(cmap_name)
    ax.imshow(raster, aspect="auto", interpolation="nearest", cmap=cmap)
    fs = 8 if compact else 10

    tick_idx = np.linspace(0, n_tau - 1, min(8, n_tau)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{result.tau_grid[i]:.2g}" for i in tick_idx],
                       rotation=30, fontsize=fs - 2)
    ax.set_xlabel(r"$\tau$", fontsize=fs)
    ax.set_ylabel("node", fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs - 2)

    if highlight_metastable and result.metastable:
        meta_nodes = sorted(result.metastable.keys())
        for n_idx in meta_nodes:
            ax.axhline(n_idx, color="red", linewidth=0.3, alpha=0.4)

    if title:
        ax.set_title(title, fontsize=fs)
    return fig, ax


def plot_tanglegram(
    result_a: MultiscaleResult,
    result_b: MultiscaleResult,
    *,
    tau_value: float | None = None,
    labels: tuple[str, str] | None = None,
    figsize: tuple[float, float] = (10.0, 4.5),
) -> matplotlib.figure.Figure:
    """Paired dendrograms with leaf-to-leaf connections.

    Draws dendrogram A on the left, B on the right (mirrored), and a
    thin line for each leaf connecting its position in A to its
    position in B. Highlights crossings — i.e. pairs of leaves whose
    relative order differs between the two dendrograms.
    """
    if not result_a.per_tau or not result_b.per_tau:
        raise ValueError("Both MultiscaleResults must have per-τ entries.")
    if tau_value is None:
        tau_value = result_a.tau_prime
    idx_a = _nearest_tau_index(result_a.tau_grid, tau_value)
    idx_b = _nearest_tau_index(result_b.tau_grid, tau_value)
    entry_a = result_a.per_tau[idx_a]
    entry_b = result_b.per_tau[idx_b]

    fig, (ax_l, ax_mid, ax_r) = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1.5, 3], "wspace": 0.02},
    )
    d_a = dendrogram(entry_a["linkage"], ax=ax_l, orientation="left",
                     no_labels=True, color_threshold=entry_a["flat_threshold"],
                     above_threshold_color="grey")
    d_b = dendrogram(entry_b["linkage"], ax=ax_r, orientation="right",
                     no_labels=True, color_threshold=entry_b["flat_threshold"],
                     above_threshold_color="grey")

    leaves_a = d_a["leaves"]
    leaves_b = d_b["leaves"]
    pos_a = {leaf: i for i, leaf in enumerate(leaves_a)}
    pos_b = {leaf: i for i, leaf in enumerate(leaves_b)}

    common = set(leaves_a) & set(leaves_b)
    n_common = len(common)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-0.5, max(len(leaves_a), len(leaves_b)) - 0.5)
    ax_mid.axis("off")
    for leaf in common:
        y_a = pos_a[leaf]
        y_b = pos_b[leaf]
        rel_diff = abs(y_a - y_b) / max(1, n_common)
        color = plt.cm.RdYlGn_r(rel_diff)
        ax_mid.plot([0, 1], [y_a, y_b], color=color, linewidth=0.5, alpha=0.7)

    ax_l.set_xticks([])
    ax_r.set_xticks([])
    ax_l.set_title(
        labels[0] if labels else (result_a.label or "A"),
        fontsize=10,
    )
    ax_r.set_title(
        labels[1] if labels else (result_b.label or "B"),
        fontsize=10,
    )
    ax_r.invert_xaxis()
    return fig


def plot_metastable_overlay(
    result: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Bar chart of how many τ-switches each metastable node experiences."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 2.5))
    else:
        fig = ax.get_figure()

    if not result.metastable:
        ax.text(0.5, 0.5, "no metastable nodes", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_axis_off()
        return fig, ax

    nodes = sorted(result.metastable.keys(),
                   key=lambda n: -result.metastable[n]["n_switches"])
    switches = [result.metastable[n]["n_switches"] for n in nodes]
    fs = 8 if compact else 10
    ax.bar(range(len(nodes)), switches, color="#6A1B9A")
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([str(int(result.nodes_alive[n])) if n < len(result.nodes_alive)
                        else str(n) for n in nodes],
                       rotation=90, fontsize=fs - 3)
    ax.set_ylabel("# switches", fontsize=fs)
    ax.set_xlabel("node (original ROI index)", fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=fs)
    return fig, ax
