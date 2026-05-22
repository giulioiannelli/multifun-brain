"""Shared scaffolding for ``plotlib`` helpers.

Every plot helper follows the same signature::

    def plot_xxx(result, *, ax=None, title=None, colorbar=True, legend=True, ...):
        fig, ax = ensure_axes(ax, figsize=...)
        # ...draw...
        apply_decorations(ax, title=title, ...)
        return fig, ax

The two helpers below capture the boilerplate so each plot file stays
focused on the actual drawing logic, and grid composers can rely on a
consistent set of decoration kwargs.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["apply_decorations", "ensure_axes"]


def ensure_axes(
    ax: Axes | None,
    *,
    figsize: tuple[float, float] | None = None,
    **subplots_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating them if ``ax`` is ``None``.

    The canonical guard used by every plot helper. Lets a caller either
    pass an existing axes (for grid composition) or omit it (for the
    one-up case). Extra ``subplots_kwargs`` are forwarded to
    :func:`matplotlib.pyplot.subplots`.

    Parameters
    ----------
    ax:
        Pre-existing axes to draw on, or ``None`` to create a new figure.
    figsize:
        Figure size for the new axes. Ignored when ``ax`` is provided.
    **subplots_kwargs:
        Forwarded to ``plt.subplots`` when creating a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **subplots_kwargs)
    else:
        fig = ax.get_figure()
    return fig, ax


def apply_decorations(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    grid: bool = False,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """Apply the standard decoration kwargs that all plot helpers accept.

    Parameters
    ----------
    ax:
        Axes to decorate.
    title:
        Axes title; pass ``None`` to omit (grid composers often do).
    xlabel, ylabel:
        Axis labels; pass ``None`` to omit.
    legend:
        Show the legend if ``True``. The legend is only added when
        artists actually have labels.
    grid:
        Toggle the grid.
    grid_kwargs, legend_kwargs:
        Optional extra kwargs forwarded to :meth:`Axes.grid` /
        :meth:`Axes.legend`.
    """
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, **(grid_kwargs or {}))
    if legend:
        # Only attach a legend if some artist on the axes has a label.
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(**(legend_kwargs or {}))
