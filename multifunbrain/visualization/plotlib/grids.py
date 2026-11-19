"""Grid composers — multi-panel layouts assembled from the section files.

``plot_results_grid`` is the canonical "same plot for N results in a
grid" helper; every notebook with batch comparison should reach for
this instead of writing a loop. ``plot_pipeline_summary`` is the
six-panel one-up overview of a single :class:`PipelineResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from .descriptive import (
    plot_correlation_matrix,
    plot_eigenvalue_spectrum,
    plot_signed_laplacian_spectrum,
    plot_weight_distribution,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_pipeline_summary",
    "plot_results_grid",
]


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
    """Render ``plot_fn`` into a ``len(row_keys) x len(col_keys)`` grid.

    Each cell ``(i, j)`` looks up
    ``results_by_key[(row_keys[i], col_keys[j])]`` and calls
    ``plot_fn(result, ax=ax, **plot_kwargs)``. Missing keys yield a
    blank "no data" cell; per-cell exceptions are caught and rendered
    as the error string so one bad subplot does not abort the figure.

    Parameters
    ----------
    results_by_key : dict[(row_key, col_key) -> PipelineResult]
        Mapping from grid coordinates to results.
    plot_fn : callable
        ``plot_fn(result, *, ax, **kwargs) -> (fig, ax)`` — any plot
        function from this package that accepts ``ax`` works.
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
        ``axes`` is always a 2D ``ndarray`` of shape
        ``(len(row_keys), len(col_keys))``.
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
