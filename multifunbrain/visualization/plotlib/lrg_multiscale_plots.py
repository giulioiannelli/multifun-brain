"""Multiscale LRG visualisation primitives.

Each helper takes a :class:`MultiscaleResult` (or a pair of them) and an
optional ``ax``. Returns ``(fig, ax)`` so it composes with
:func:`plot_results_grid`. All decorations are kept compact so the same
helper renders cleanly both as a stand-alone figure and as a sub-panel.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from ...analysis.lrg.compare import (
    reduced_mutual_information,
    specific_heat_comparison,
)
from ...pipeline.multiscale import MultiscaleResult

__all__ = [
    "plot_specific_heat",
    "plot_specific_heat_overlay",
    "plot_dendrogram_with_psi",
    "plot_psi_curve",
    "plot_rmi_curve",
    "plot_partition_flow",
    "plot_tanglegram",
    "plot_metastable_overlay",
]


def _nearest_tau_index(tau_grid: np.ndarray, tau_value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(tau_grid) - tau_value)))


def plot_specific_heat(
    result: MultiscaleResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    mark_tau_prime: bool = True,
    mark_tau_star: bool = True,
    title: str | None = None,
    color: str = "#1565C0",
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """``C(τ)`` curve with τ′ and τ* marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
    else:
        fig = ax.get_figure()

    ax.semilogx(result.time_grid_broad, result.specific_heat, "-",
                color=color, linewidth=1.4)
    ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)

    if mark_tau_prime and np.isfinite(result.tau_prime):
        ax.axvline(result.tau_prime, color="#E65100", linestyle="--",
                   linewidth=1.0, alpha=0.8, label=r"$\tau'=1/\lambda_{max}$")
    if mark_tau_star and np.isfinite(result.tau_star):
        ax.axvline(result.tau_star, color="#2E7D32", linestyle=":",
                   linewidth=1.2, alpha=0.9, label=r"$\tau^*$")

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

    Each row is a node, each column a τ, and the cell colour encodes the
    cluster label at that τ. Metastable nodes (those whose membership-set
    changes) are marked with a vertical red tick at the row's far right.
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

    Draws dendrogram A on the left, B on the right (mirrored), and a thin
    line for each leaf connecting its position in A to its position in B.
    Highlights crossings — i.e. pairs of leaves whose relative order
    differs between the two dendrograms.
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
