"""Centralised plotting style: palettes, figsizes, rcParams profiles.

This is the single source of truth for the colours and sizes the
``multifunbrain.visualization.plotlib`` helpers use. Plot functions
should pull hex strings from :data:`PALETTES` rather than hardcoding
them so that re-skinning the whole library is a one-file change.

Usage::

    from multifunbrain.visualization.style import PALETTES, FIGSIZE, rc_context

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    ax.plot(x, y, color=PALETTES["material"]["primary"])

    with rc_context("paper"):
        fig = make_my_paper_figure(...)

Adding a palette or profile: extend the dicts below — every plot
helper that uses them picks up the change automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt

__all__ = ["FIGSIZE", "PALETTES", "RC_PROFILES", "rc_context"]


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

#: Material Design accent colours — primary library palette.
_MATERIAL = {
    "primary": "#1976D2",      # blue 700
    "primary_dark": "#1565C0", # blue 800
    "primary_light": "#2196F3",# blue 500
    "accent": "#F57C00",       # orange 700
    "accent_light": "#FF9800", # orange 500
    "danger": "#D32F2F",       # red 700
    "danger_light": "#F44336", # red 500
    "success": "#388E3C",      # green 700
    "success_light": "#4CAF50",# green 500
    "neutral": "#616161",      # grey 700
}

#: 13-colour qualitative palette suitable for partition / module colouring.
_SPECTRAL = [
    "#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
    "#FFFFBF", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD",
    "#5E4FA2", "#3498DB", "#27AE60",
]

#: Two-colour pair for signed networks (positive / negative edges).
_SIGNED = {"pos": "#27AE60", "neg": "#E74C3C"}

#: Two-colour pair for contrast comparisons used in the April batch
#: (hypercapnia vs. resting state).
_CONTRAST = {"co2": "#D32F2F", "rest": "#1976D2"}

PALETTES: dict[str, Any] = {
    "material": _MATERIAL,
    "spectral": _SPECTRAL,
    "signed": _SIGNED,
    "contrast": _CONTRAST,
}


# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------

#: Standard figure sizes (inches). Use them as the ``figsize=`` argument so
#: notebooks render uniformly and grids fit on a page without manual tuning.
FIGSIZE: dict[str, tuple[float, float]] = {
    "single": (6.0, 4.0),       # one-up panel
    "wide": (10.0, 4.0),        # time-series / spectrum band
    "tall": (4.0, 6.0),         # bar charts / dendrograms
    "square": (5.0, 5.0),       # adjacency / correlation heatmap
    "grid_cell": (2.5, 2.5),    # cell of a results grid
    "double": (12.0, 4.0),      # two side-by-side panels
}


# ---------------------------------------------------------------------------
# rcParams profiles
# ---------------------------------------------------------------------------

#: Named rcParams overrides. ``paper`` is tuned for publication PDFs
#: (serif, 8 pt base, hairline axes). ``slide`` is tuned for projector
#: viewing (sans-serif, larger fonts).
RC_PROFILES: dict[str, dict[str, Any]] = {
    "paper": {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
    },
    "slide": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "savefig.dpi": 150,
    },
}


@contextmanager
def rc_context(profile: str = "paper"):
    """Apply a named rcParams profile inside a ``with`` block.

    The change reverts when the block exits — safe for one-off figures
    inside a notebook that should otherwise keep matplotlib defaults.

    Parameters
    ----------
    profile:
        Name from :data:`RC_PROFILES` (currently ``"paper"`` or
        ``"slide"``).

    Examples
    --------
    >>> with rc_context("paper"):
    ...     fig, ax = plt.subplots()
    ...     ax.plot([0, 1], [0, 1])
    """
    if profile not in RC_PROFILES:
        raise ValueError(
            f"Unknown rc profile {profile!r}. Available: {sorted(RC_PROFILES)}."
        )
    with plt.rc_context(RC_PROFILES[profile]):
        yield
