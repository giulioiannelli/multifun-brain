"""Plotting functions for pipeline results.

Each function follows a consistent pattern:
- Accepts a :class:`~multifunbrain.pipeline.PipelineResult` (or the
  relevant sub-dict) as the first argument.
- Accepts an optional ``ax`` (or ``axes``) parameter.  When *None* a
  new figure is created; otherwise the provided axes are drawn into.
- Returns ``(fig, ax)`` so callers can further customise or save.

Typical usage::

    from multifunbrain.pipeline import load_results
    from multifunbrain.visualization.plotlib.pipeline_plots import (
        plot_weight_distribution,
        plot_eigenvalue_spectrum,
        plot_pipeline_summary,
    )

    results = load_results("pipeline_results/")
    r = results[0]

    fig, ax = plot_weight_distribution(r)
    fig, ax = plot_eigenvalue_spectrum(r)
    fig, axes = plot_pipeline_summary(r)   # multi-panel overview
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_correlation_matrix",
    "plot_eigenvalue_spectrum",
    "plot_filtered_comparison",
    "plot_lrg_dendrogram",
    "plot_lrg_entropy",
    "plot_lrg_partition_network",
    "plot_lrg_psi",
    "plot_lrg_sankey",
    "plot_network",
    "plot_node_metrics",
    "plot_percolation_curve",
    "plot_pipeline_summary",
    "plot_results_grid",
    "plot_signed_balance",
    "plot_signed_laplacian_spectrum",
    "plot_signed_network",
    "plot_weight_distribution",
]


# ──────────────────────────────────────────────────────────────────────
# Grid helper — reusable multi-panel layout
# ──────────────────────────────────────────────────────────────────────


def plot_results_grid(
    results_by_key: dict[tuple[Any, Any], PipelineResult],
    plot_fn,
    row_keys: list,
    col_keys: list,
    *,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3.5, 3.0),
    row_label: str | None = None,
    col_label: str | None = None,
    suptitle: str | None = None,
    sharex: bool = False,
    sharey: bool = False,
    **plot_kwargs,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Render *plot_fn* into a ``len(row_keys) x len(col_keys)`` grid.

    Each cell ``(i, j)`` looks up ``results_by_key[(row_keys[i], col_keys[j])]``
    and calls ``plot_fn(result, ax=ax, **plot_kwargs)``. Missing keys yield a
    blank "no data" cell; per-cell exceptions are caught and rendered as the
    error string so one bad subplot does not abort the figure.

    Parameters
    ----------
    results_by_key : dict[(row_key, col_key) -> PipelineResult]
        Mapping from grid coordinates to results.
    plot_fn : callable
        ``plot_fn(result, *, ax, **kwargs) -> (fig, ax)`` — any plot function
        from this module that accepts ``ax`` works.
    row_keys, col_keys : list
        Outer / inner grid keys, in the order they should appear.
    figsize_per_cell : (float, float)
        Width × height in inches of each subplot.
    row_label, col_label : str or None
        Optional axis labels printed on the leftmost column / top row.
    suptitle : str or None
        Optional figure-level title.
    sharex, sharey : bool
        Forwarded to :func:`plt.subplots`.
    **plot_kwargs
        Extra keyword arguments forwarded to ``plot_fn``.

    Returns
    -------
    (fig, axes)
        ``axes`` is always a 2D ``ndarray`` of shape ``(len(row_keys), len(col_keys))``.
    """
    nrows, ncols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_cell[0], nrows * figsize_per_cell[1]),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            ax = axes[i, j]
            r = results_by_key.get((rk, ck))
            if r is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                try:
                    plot_fn(r, ax=ax, **plot_kwargs)
                except Exception as exc:
                    ax.clear()
                    ax.text(0.5, 0.5, f"{type(exc).__name__}:\n{exc}",
                            ha="center", va="center", transform=ax.transAxes,
                            color="#c62828", fontsize=7, wrap=True)
                    ax.set_xticks([])
                    ax.set_yticks([])

            if i == 0:
                col_text = col_labels[j] if col_labels is not None else str(ck)
                ax.set_title(col_text, fontsize=9)
            if j == 0:
                row_text = row_labels[i] if row_labels is not None else str(rk)
                if row_label is not None:
                    row_text = f"{row_label}={row_text}"
                ax.set_ylabel(row_text, fontsize=9)

    if col_label is not None:
        fig.supxlabel(col_label, fontsize=10)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────────────────────────────
# Percolation curves — exploratory backbone visualisation
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Section 1: Descriptive analysis plots
# ──────────────────────────────────────────────────────────────────────


def plot_weight_distribution(
    result: PipelineResult | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
    n_bins: int = 80,
    colors: tuple[str, str] = ("#2196F3", "#F44336"),
    legend: bool = True,
    title: str | None = "Weight distribution",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Histogram of correlation weights split into positive / negative.

    Parameters
    ----------
    result : PipelineResult or dict
        Either a full pipeline result or its ``descriptive["weight_distribution"]`` dict.
    ax : Axes or None
        Target axes.
    n_bins : int
        Number of histogram bins.
    colors : tuple of str
        ``(positive_color, negative_color)``.

    Returns
    -------
    (fig, ax)
    """
    wd = _get_section(result, "weight_distribution")
    pos = np.asarray(wd["positive_weights"])
    neg = np.asarray(wd["negative_weights"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    all_w = np.concatenate([neg, pos]) if len(neg) and len(pos) else (pos if len(pos) else neg)
    bins = np.linspace(all_w.min(), all_w.max(), n_bins + 1)

    if len(pos):
        ax.hist(pos, bins=bins, alpha=0.7, color=colors[0], label=f"positive ({len(pos)})")
    if len(neg):
        ax.hist(neg, bins=bins, alpha=0.7, color=colors[1], label=f"negative ({len(neg)})")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Correlation weight")
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_eigenvalue_spectrum(
    result: PipelineResult | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_mp: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Eigenvalue spectrum of the correlation matrix, optionally with MP bounds.

    Parameters
    ----------
    result : PipelineResult or dict
        Full pipeline result or its ``descriptive["spectrum"]`` dict.
    ax : Axes or None
        Target axes.
    show_mp : bool
        Overlay Marchenko-Pastur bounds when available.

    Returns
    -------
    (fig, ax)
    """
    spec = _get_section(result, "spectrum")
    eigenvalues = np.sort(np.asarray(spec["eigenvalues"]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, "o-", markersize=3,
            color="#1565C0", label="Eigenvalues")

    if show_mp and "mp_lambda_plus" in spec:
        lp = spec["mp_lambda_plus"]
        lm = spec["mp_lambda_minus"]
        ax.axhspan(lm, lp, alpha=0.15, color="#FF9800", label=f"MP bulk [{lm:.2f}, {lp:.2f}]")
        n_sig = spec.get("n_signal", 0)
        ax.set_title(f"Eigenvalue spectrum ({n_sig} signal eigenvalues)")
    else:
        ax.set_title("Eigenvalue spectrum")

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_signed_laplacian_spectrum(
    result: PipelineResult | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Signed Laplacian eigenvalue spectrum, highlighting negative eigenvalues.

    Returns
    -------
    (fig, ax)
    """
    sl = _get_section(result, "signed_laplacian")
    eigenvalues = np.sort(np.asarray(sl["eigenvalues"]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    colours = np.where(eigenvalues < 0, "#F44336", "#4CAF50")
    ax.bar(np.arange(len(eigenvalues)), eigenvalues, color=colours, width=0.8)
    ax.axhline(0, color="k", linewidth=0.8)

    n_neg = int(sl.get("n_negative_eigenvalues", (eigenvalues < 0).sum()))
    frust = sl.get("frustration_index", 0.0)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Signed Laplacian spectrum ({n_neg} negative, frustration={frust:.3f})")
    fig.tight_layout()
    return fig, ax


def plot_signed_balance(
    result: PipelineResult,
    *,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Signed network balance profile.

    Left panel: scatter of positive-strength vs negative-strength per
    node.  The diagonal represents perfect balance (equal positive and
    negative weight).  Points above the diagonal are "antagonistic"
    nodes (more negative connections).

    Right panel: histogram of per-node balance ratio
    ``s+/(s+ + s−)``.  Values near 1 = mostly positive connections,
    near 0.5 = balanced, near 0 = mostly negative.

    Returns
    -------
    (fig, axes)
    """
    sm = _get_section(result, "network_metrics")
    s_pos = np.asarray(sm["strength_positive"])
    s_neg = np.asarray(sm["strength_negative"])
    s_total = s_pos + s_neg
    balance = np.where(s_total > 0, s_pos / s_total, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: scatter
    ax = axes[0]
    ax.scatter(s_pos, s_neg, s=20, alpha=0.7, c=balance, cmap="RdYlBu",
               edgecolors="k", linewidths=0.3, vmin=0, vmax=1)
    lim = max(s_pos.max(), s_neg.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Positive strength")
    ax.set_ylabel("Negative strength")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    # Right: balance histogram
    ax = axes[1]
    ax.hist(balance, bins=30, color="#5C6BC0", edgecolor="k",
            linewidth=0.5, alpha=0.8)
    ax.axvline(0.5, color="#F44336", linestyle="--", linewidth=1,
               label="perfect balance")
    mean_b = float(np.mean(balance))
    ax.axvline(mean_b, color="#FF9800", linestyle="-", linewidth=1,
               label=f"mean={mean_b:.2f}")
    ax.set_xlabel("Balance ratio  s⁺/(s⁺+s⁻)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_correlation_matrix(
    result: PipelineResult | np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    colorbar: bool = True,
    vmax: float | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Heatmap of the (prepared) correlation matrix with diverging colormap.

    Parameters
    ----------
    result : PipelineResult or np.ndarray
        Full result (uses ``corr_prepared``) or a raw matrix.
    title : str or None
        Axes title.  *None* (default) suppresses the title.

    Returns
    -------
    (fig, ax)
    """
    if isinstance(result, np.ndarray):
        matrix = result
    else:
        matrix = result.corr_prepared

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    if vmax is None:
        vmax = float(np.max(np.abs(matrix)))
    if vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    if title:
        ax.set_title(title)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Section 2: Filtering comparison
# ──────────────────────────────────────────────────────────────────────


def plot_filtered_comparison(
    result: PipelineResult,
    *,
    figsize: tuple[float, float] = (18, 6),
    k: float = 0.08,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Three-panel network comparison: absolute, positive, full signed.

    Left panel:  absolute-value network (thresholded).
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
        ("absolute", "#546E7A", None),   # name, edge colour, None = unsigned
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

        # Layout: spring with small k, topology-only (weight=None) to
        # avoid heavy links compressing the graph.
        pos = nx.spring_layout(G, k=k, seed=42, iterations=200, weight=None)

        # Edge drawing: width & alpha proportional to weight.
        wts = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        wmax = wts.max() if len(wts) else 1.0
        widths = wts / wmax * 2.0
        alphas = wts / wmax * 0.7 + 0.05

        # Draw edges one by one for per-edge alpha.
        for (u, v, d), w, a in zip(G.edges(data=True), widths, alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=edge_col, alpha=float(a),
            )

        # Nodes: uniform small circles.
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

        # Build edge lists.
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

        # Draw edges with per-edge alpha.
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


# ──────────────────────────────────────────────────────────────────────
# Section 3: Node metrics
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Multi-panel summary
# ──────────────────────────────────────────────────────────────────────


def plot_pipeline_summary(
    result: PipelineResult,
    *,
    figsize: tuple[float, float] = (16, 10),
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Six-panel overview of a single pipeline result.

    Layout (2 rows x 3 cols)::

        [corr matrix]  [weight distribution]  [eigenvalue spectrum]
        [signed Lapl]  [filter comparison]    [node metrics]

    Returns
    -------
    (fig, axes)
        2x3 array of axes.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    title = result.label or "Pipeline result"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Row 1
    plot_correlation_matrix(result, ax=axes[0, 0], title="Correlation matrix")
    plot_weight_distribution(result, ax=axes[0, 1])
    plot_eigenvalue_spectrum(result, ax=axes[0, 2])

    # Row 2
    plot_signed_laplacian_spectrum(result, ax=axes[1, 0])

    if result.network_analyses:
        ax = axes[1, 1]
        filters = list(result.network_analyses.keys())
        densities = [
            result.network_analyses[f].get("global_metrics", {}).get("density", 0.0)
            for f in filters
        ]
        colors = ["#546E7A", "#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]
        ax.bar(filters, densities, color=colors[: len(filters)])
        ax.set_ylabel("Density")
        ax.set_title("Filtered network density")
        ax.tick_params(axis="x", rotation=20)

        # Node metrics for the first filter — single metric on the last panel
        first_filter = next(iter(result.network_analyses))
        analysis = result.network_analyses[first_filter]
        node_df = analysis.get("node_metrics")
        if node_df is not None and "strength" in node_df.columns:
            ax = axes[1, 2]
            vals = node_df["strength"].dropna().values
            ax.hist(vals, bins=20, color="#5C6BC0", edgecolor="k", alpha=0.8)
            ax.axvline(np.mean(vals), color="#F44336", linestyle="--",
                       label=f"mean={np.mean(vals):.3f}")
            ax.set_xlabel("Strength")
            ax.set_ylabel("Count")
            ax.set_title(f"Node strength ({first_filter})")
            ax.legend(fontsize=8)
        else:
            axes[1, 2].set_visible(False)
    else:
        axes[1, 1].text(0.5, 0.5, "No filtered\nnetworks", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=12, color="gray")
        axes[1, 2].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _get_section(result: Any, key: str) -> dict[str, Any]:
    """Extract a descriptive sub-dict from a PipelineResult or raw dict."""
    if isinstance(result, dict):
        # Already the sub-dict
        if key in result:
            return result[key]
        return result
    # PipelineResult
    desc = getattr(result, "descriptive", None)
    if desc is None:
        raise ValueError("Result has no descriptive analysis. Run the pipeline first.")
    if key in desc:
        return desc[key]
    raise KeyError(f"Key {key!r} not found in descriptive results. "
                   f"Available: {list(desc.keys())}")


def _resolve_filter(result: Any, filter_name: str | None) -> str:
    """Pick a filter name, defaulting to the first available."""
    if filter_name is not None:
        return filter_name
    if result.filtered_networks:
        return next(iter(result.filtered_networks))
    raise ValueError("No filtered networks available in this result.")


def _signed_laplacian_embedding(corr: np.ndarray) -> dict:
    """2-D spectral embedding from the signed Laplacian.

    The two eigenvectors of L_s = |D| − A associated with the smallest
    eigenvalues capture the dominant bipartition of the signed network,
    placing nodes in the same "balanced community" close together.
    """
    C = corr.copy()
    np.fill_diagonal(C, 0.0)
    abs_C = np.abs(C)
    D = np.diag(abs_C.sum(axis=1))
    L_s = D - C

    eigenvalues, eigenvectors = np.linalg.eigh(L_s)
    # Smallest eigenvalues capture the frustrated bipartition.
    idx = np.argsort(eigenvalues)
    coords = eigenvectors[:, idx[:2]]

    # Normalise to unit square for consistent rendering.
    for dim in range(2):
        r = coords[:, dim].max() - coords[:, dim].min()
        if r > 0:
            coords[:, dim] = (coords[:, dim] - coords[:, dim].min()) / r

    return {i: coords[i] for i in range(len(coords))}


def _network_layout(G: nx.Graph, signed: bool = False) -> dict:
    """Compute a layout that reveals modular structure.

    For unsigned networks (all positive weights), uses Kamada-Kawai with
    distance = 1/weight so strongly connected nodes are placed close.
    For signed networks, uses spring layout where positive weights attract
    and negative weights repel.
    """
    if G.number_of_nodes() == 0:
        return {}
    if signed:
        return nx.spring_layout(G, weight="weight", seed=42, iterations=100)
    # Kamada-Kawai needs a distance matrix; invert weights so strong = close.
    try:
        return nx.kamada_kawai_layout(G, weight="weight")
    except (ValueError, nx.NetworkXError):
        return nx.spring_layout(G, weight="weight", seed=42, iterations=100)


# ──────────────────────────────────────────────────────────────────────
# Network layout plots
# ──────────────────────────────────────────────────────────────────────


def plot_signed_network(
    result: PipelineResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw the raw signed correlation network.

    Positive edges in blue, negative in red.  Edge width/alpha scale
    with |weight|.  Nodes are colored by the Louvain partition of the
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

    # Node colors from absolute-filter community if available.
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

    # Separate positive / negative edges.
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

    # Community colors.
    analysis = result.network_analyses.get(fname, {})
    partition = analysis.get("community", {}).get("partition", {})
    cmap = plt.cm.tab20
    node_colors = [cmap(partition.get(n, 0) % 20) for n in nodes]

    # Node sizes from strength.
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


# ──────────────────────────────────────────────────────────────────────
# LRG multiscale plots
# ──────────────────────────────────────────────────────────────────────


def plot_lrg_entropy(
    result: PipelineResult,
    filter_name: str | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Entropy (1-S) and specific heat (C) on dual y-axes.

    Recomputes the Laplacian spectrum from the stored filtered graph,
    then calls the existing ``entropy()`` function from lrglib.

    Returns
    -------
    (fig, ax)
    """
    from ...analysis.lrglib import entropy as lrg_entropy
    from ...analysis.lrglib import graph_laplacian_and_spectrum

    fname = _resolve_filter(result, filter_name)
    G = result.filtered_networks[fname]["graph"]
    _, spectrum = graph_laplacian_and_spectrum(G, normalized=True)

    Sm1, dS, _varL, t = lrg_entropy(spectrum)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    from .entropy import plot_entropy_and_C
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

    Draws the dendrogram for the partition at *tau_index* and a
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
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    dendrogram(Z, ax=ax, color_threshold=threshold, leaf_font_size=6)
    ax.axhline(threshold, color="#F44336", linestyle="--", linewidth=1.2,
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
    from ...analysis.lrglib import compute_optimal_threshold

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

    Uses the partition at *tau_index* from the LRG results for the
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

    Reuses the existing ``plot_sankey_matplotlib`` function.

    Returns
    -------
    (fig, ax)
    """
    from .sankey_matplotlib import plot_sankey_matplotlib

    fname = _resolve_filter(result, filter_name)
    parts = result.lrg_results.get(fname)
    if not parts:
        raise ValueError(f"No LRG results for filter '{fname}'.")

    partitions = [p["partition"] for p in parts]
    tau_values = [p["tau"] for p in parts]

    # plot_sankey_matplotlib creates its own figure internally.
    plot_sankey_matplotlib(partitions, tau_values)
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None
    fig.suptitle(f"LRG Sankey ({fname})", fontsize=12)
    return fig, ax
