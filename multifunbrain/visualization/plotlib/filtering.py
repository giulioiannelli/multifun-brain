"""Section 2 plots — network filtering comparisons.

Plots that visualise the effect of filters that turn the signed dense
correlation matrix into one or more unsigned networks: percolation
curves, and a three-panel before/after comparison of representative
filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._helpers import _signed_laplacian_embedding

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_filtered_comparison",
    "plot_percolation_curve",
]


def plot_percolation_curve(
    result: PipelineResult | np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    n_thresholds: int = 40,
    show_e_inf: bool = True,
    title: str | None = None,
    colors: tuple[str, str] = ("#1565C0", "#E65100"),
    compact: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot ``P_∞`` and (optionally) ``E_∞`` versus the ``|corr|`` threshold.

    Reveals the percolation structure of the correlation network:
    a staircase in ``P_∞`` suggests functional submodules; a single
    sharp drop suggests random-graph-like topology.

    Parameters
    ----------
    result : PipelineResult or np.ndarray
        Either a full result (uses ``corr_prepared``) or a raw matrix.
    ax : Axes or None
        Target axes (composes into a grid when provided).
    n_thresholds : int
        Number of threshold steps.
    show_e_inf : bool
        Overlay ``E_∞`` on a twin axis. Disable for very small panels.
    title : str or None
        Axes title.
    colors : (str, str)
        ``(P_∞ color, E_∞ color)``.
    compact : bool
        If True (default), use short axis labels suitable for grid cells.

    Returns
    -------
    (fig, ax)
    """
    from ...processing.percolation import percolation_curve

    if isinstance(result, np.ndarray):
        matrix = result
    else:
        matrix = result.corr_prepared
    if matrix is None:
        raise ValueError("No correlation matrix available on result.")

    weights = np.abs(matrix)
    curve = percolation_curve(weights, n_thresholds=n_thresholds, compute_e_inf=show_e_inf)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.plot(curve["thresholds"], curve["p_inf"], "-", color=colors[0], linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    if compact:
        ax.set_xlabel("Th", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    else:
        ax.set_xlabel("threshold |corr|")
    ax.set_ylabel(r"$P_\infty$", color=colors[0], fontsize=8 if compact else 10)
    ax.tick_params(axis="y", labelcolor=colors[0])

    if show_e_inf:
        ax2 = ax.twinx()
        ax2.plot(curve["thresholds"], curve["e_inf"], "-", color=colors[1], linewidth=1.2)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel(r"$E_\infty$", color=colors[1], fontsize=8 if compact else 10)
        ax2.tick_params(axis="y", labelcolor=colors[1], labelsize=7 if compact else 9)

    if title:
        ax.set_title(title, fontsize=8 if compact else 10)
    return fig, ax


def plot_filtered_comparison(
    result: PipelineResult,
    *,
    figsize: tuple[float, float] = (18, 6),
    k: float = 0.08,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Three-panel network comparison: absolute, positive, full signed.

    Left panel:   absolute-value network (thresholded).
    Centre panel: positive-only network (thresholded).
    Right panel:  full signed network (positive=blue, negative=red)
                  laid out with signed-Laplacian spectral embedding.

    Edge widths and transparency scale with ``|weight|``:
    weight 0 is fully transparent, ``±max`` is the thickest.

    Returns
    -------
    (fig, axes)
        Array of 3 axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    panels = [
        ("absolute", "#546E7A", None),
        ("positive", "#1565C0", None),
    ]

    # ── Unsigned panels (absolute, positive) ──────────────────────────
    for (fname, edge_col, _), ax in zip(panels, axes[:2]):
        fdata = result.filtered_networks.get(fname)
        if fdata is None:
            ax.text(0.5, 0.5, f"No '{fname}' filter", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")
            continue

        G = fdata["graph"]
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            continue

        pos = nx.spring_layout(G, k=k, seed=42, iterations=200, weight=None)

        wts = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        wmax = wts.max() if len(wts) else 1.0
        widths = wts / wmax * 2.0
        alphas = wts / wmax * 0.7 + 0.05

        for (u, v, _d), w, a in zip(G.edges(data=True), widths, alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=edge_col, alpha=float(a),
            )

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=25, node_color="#37474F",
            edgecolors="k", linewidths=0.2,
        )

        perc = fdata.get("percolation", {})
        th_str = f", Th={perc.get('p_inf', 0):.2f}" if perc else ""
        ax.set_title(
            f"{fname}  ({G.number_of_nodes()} n, "
            f"{G.number_of_edges()} e{th_str})",
            fontsize=10,
        )
        ax.axis("off")

    # ── Signed panel (right) ──────────────────────────────────────────
    ax_signed = axes[2]
    corr = result.corr_prepared
    if corr is not None:
        pos_layout = _signed_laplacian_embedding(corr)
        n = corr.shape[0]

        pos_edges, neg_edges = [], []
        pos_w, neg_w = [], []
        for i in range(n):
            for j in range(i + 1, n):
                w = corr[i, j]
                if w > 0:
                    pos_edges.append((i, j))
                    pos_w.append(w)
                elif w < 0:
                    neg_edges.append((i, j))
                    neg_w.append(abs(w))

        wmax = max(
            max(pos_w) if pos_w else 0,
            max(neg_w) if neg_w else 0,
            1e-12,
        )

        G_signed = nx.Graph()
        G_signed.add_nodes_from(range(n))
        for i, j in pos_edges:
            G_signed.add_edge(i, j)
        for i, j in neg_edges:
            G_signed.add_edge(i, j)

        for (u, v), w in zip(pos_edges, pos_w):
            a = w / wmax * 0.6 + 0.02
            nx.draw_networkx_edges(
                G_signed, pos_layout, edgelist=[(u, v)], ax=ax_signed,
                width=w / wmax * 2.0, edge_color="#1565C0", alpha=float(a),
            )
        for (u, v), w in zip(neg_edges, neg_w):
            a = w / wmax * 0.6 + 0.02
            nx.draw_networkx_edges(
                G_signed, pos_layout, edgelist=[(u, v)], ax=ax_signed,
                width=w / wmax * 2.0, edge_color="#C62828", alpha=float(a),
            )

        nx.draw_networkx_nodes(
            G_signed, pos_layout, ax=ax_signed, node_size=25,
            node_color="#37474F", edgecolors="k", linewidths=0.2,
        )
        n_pos = len(pos_edges)
        n_neg = len(neg_edges)
        ax_signed.set_title(
            f"signed  ({n} n, {n_pos}+ / {n_neg}−)",
            fontsize=10,
        )
    else:
        ax_signed.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax_signed.transAxes, color="gray")

    ax_signed.axis("off")
    fig.tight_layout()
    return fig, axes
