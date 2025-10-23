"""Plotting utilities for :mod:`multifunbrain`.

This package centralises heavy plotting imports so that submodules can
reuse shared handles (for example ``plt`` or ``go``) without repeating
the import cost in every notebook run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D

from .colorbars import imshow_colorbar_caxdivider
from .entropy import plot_entropy_and_C
from .sankey_matplotlib import plot_sankey_matplotlib
from .sankey_plotly import plot_sankey

__all__ = [
    "plot_entropy_and_C",
    "plot_sankey_matplotlib",
    "plot_sankey",
    "plt",
    "go",
    "Rectangle",
    "Polygon",
    "Axes3D",
    "BoundaryNorm",
    "LinearSegmentedColormap",
    "ListedColormap",
    "TwoSlopeNorm",
    "imshow_colorbar_caxdivider",
]
