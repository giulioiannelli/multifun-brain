"""Plots related to entropy metrics."""

from __future__ import annotations

from matplotlib.axes import Axes


def plot_entropy_and_C(
    ax1: Axes,
    t,
    Sm1,
    Csp,
    color1: str = "blue",
    color2: str = "red",
) -> None:
    """
    Plot 1-S and C on the same x-axis (log scale) with twin y-axes.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Primary axis to plot 1-S.

    t : array-like
        Time values (must match Sm1 and Csp).

    Sm1 : array-like
        Entropy-related quantity to plot as 1-S.

    Csp : array-like
        Clustering coefficient or similar metric (must be len(t) - 1).

    color1 : str, optional
        Color for 1-S plot (default is "blue").

    color2 : str, optional
        Color for C plot (default is "red").
    """
    ax1.plot(t, Sm1, label=r"$1-S$", color=color1)
    ax1.set_ylabel("1-S", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(t[:-1], Csp, label=r"$C$", color=color2)
    ax2.set_ylabel("C", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xlabel(r"$\tau$")
