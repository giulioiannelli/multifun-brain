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
from .lrg_multiscale_plots import (
    plot_dendrogram_with_psi,
    plot_metastable_overlay,
    plot_partition_flow,
    plot_psi_curve,
    plot_rmi_curve,
    plot_specific_heat,
    plot_specific_heat_overlay,
    plot_tanglegram,
)
from .pipeline_plots import (
    plot_correlation_matrix,
    plot_eigenvalue_spectrum,
    plot_filtered_comparison,
    plot_lrg_dendrogram,
    plot_lrg_entropy,
    plot_lrg_partition_network,
    plot_lrg_psi,
    plot_lrg_sankey,
    plot_network,
    plot_node_metrics,
    plot_percolation_curve,
    plot_pipeline_summary,
    plot_results_grid,
    plot_signed_balance,
    plot_signed_laplacian_spectrum,
    plot_signed_network,
    plot_weight_distribution,
)
from .sankey_matplotlib import plot_sankey_matplotlib
from .sankey_plotly import plot_sankey

__all__ = [
    "plot_correlation_matrix",
    "plot_dendrogram_with_psi",
    "plot_eigenvalue_spectrum",
    "plot_entropy_and_C",
    "plot_filtered_comparison",
    "plot_lrg_dendrogram",
    "plot_lrg_entropy",
    "plot_lrg_partition_network",
    "plot_lrg_psi",
    "plot_lrg_sankey",
    "plot_metastable_overlay",
    "plot_network",
    "plot_node_metrics",
    "plot_partition_flow",
    "plot_percolation_curve",
    "plot_pipeline_summary",
    "plot_psi_curve",
    "plot_results_grid",
    "plot_rmi_curve",
    "plot_sankey",
    "plot_sankey_matplotlib",
    "plot_signed_balance",
    "plot_signed_laplacian_spectrum",
    "plot_signed_network",
    "plot_specific_heat",
    "plot_specific_heat_overlay",
    "plot_tanglegram",
    "plot_weight_distribution",
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
