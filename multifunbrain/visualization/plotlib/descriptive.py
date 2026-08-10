"""Section 1 plots — descriptive analysis of the raw signed network.

Plots that take a :class:`~multifunbrain.pipeline.PipelineResult` (or
the relevant sub-dict / raw matrix) and visualise weight distribution,
eigenvalue spectrum, signed-Laplacian metrics, and the correlation
matrix itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ..style import FIGSIZE
from ._helpers import _get_section

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from ...pipeline import PipelineResult

__all__ = [
    "plot_correlation_matrix",
    "plot_eigenvalue_spectrum",
    "plot_signed_balance",
    "plot_signed_laplacian_spectrum",
    "plot_weight_distribution",
]


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
    node. The diagonal represents perfect balance (equal positive and
    negative weight). Points above the diagonal are "antagonistic"
    nodes (more negative connections).

    Right panel: histogram of per-node balance ratio
    ``s+/(s+ + s−)``. Values near 1 = mostly positive connections,
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
        Axes title. ``None`` (default) suppresses the title.

    Returns
    -------
    (fig, ax)
    """
    if isinstance(result, np.ndarray):
        matrix = result
    else:
        matrix = result.corr_prepared

    if ax is None:
        # Template: CORRELATION_HEATMAP uses the "square" figure size so
        # the matrix renders with equal aspect on a sensible canvas.
        fig, ax = plt.subplots(figsize=FIGSIZE["square"])
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
