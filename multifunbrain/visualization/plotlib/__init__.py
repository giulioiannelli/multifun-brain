"""Plotting utilities for :mod:`multifunbrain`.

This package centralises the heavy matplotlib / plotly imports so that
section files can reuse shared handles (``plt``, ``go``, ``Polygon``,
``Rectangle``) without paying the import cost in each notebook cell.

Section files (one per pipeline concept) hold the actual plot functions:

- :mod:`~multifunbrain.visualization.plotlib.descriptive` — weight
  distribution, eigenvalue spectrum, signed Laplacian, correlation
  matrix heatmap.
- :mod:`~multifunbrain.visualization.plotlib.filtering` — percolation
  curves, filtered network comparison.
- :mod:`~multifunbrain.visualization.plotlib.network` — node metrics,
  signed/unsigned network drawings.
- :mod:`~multifunbrain.visualization.plotlib.lrg` — entropy/specific
  heat, dendrograms, PSI/RMI, partition flow, tanglegram, metastable
  overlay.
- :mod:`~multifunbrain.visualization.plotlib.grids` — ``plot_results_grid``
  and ``plot_pipeline_summary``.
- :mod:`~multifunbrain.visualization.plotlib.sankey` — Sankey diagrams
  (plotly + matplotlib backends behind one entry point).
- :mod:`~multifunbrain.visualization.plotlib.entropy` — dual-axis
  entropy + C atomic helper.
- :mod:`~multifunbrain.visualization.plotlib.colorbars` — colorbar
  layout utilities.
- :mod:`~multifunbrain.visualization.plotlib.base` — ``ensure_axes``
  and ``apply_decorations`` boilerplate helpers used by all plots.

Style defaults (palettes, figsizes, rcParams profiles) live one level
up in :mod:`multifunbrain.visualization.style` — pull colour and size
constants from there rather than hardcoding hex strings.
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

from .base import apply_decorations, ensure_axes
from .colorbars import imshow_colorbar_caxdivider
from .descriptive import (
    plot_correlation_matrix,
    plot_eigenvalue_spectrum,
    plot_signed_balance,
    plot_signed_laplacian_spectrum,
    plot_weight_distribution,
)
from .entropy import plot_entropy_and_C
from .filtering import plot_filtered_comparison, plot_percolation_curve
from .grids import plot_pipeline_summary, plot_results_grid
from .lrg import (
    plot_dendrogram_with_psi,
    plot_lrg_dendrogram,
    plot_lrg_entropy,
    plot_lrg_partition_network,
    plot_lrg_psi,
    plot_lrg_sankey,
    plot_metastable_overlay,
    plot_partition_flow,
    plot_psi_curve,
    plot_rmi_curve,
    plot_specific_heat,
    plot_specific_heat_overlay,
    plot_tanglegram,
)
from .network import plot_network, plot_node_metrics, plot_signed_network
from .sankey import plot_sankey, plot_sankey_matplotlib

__all__ = [
    # Public matplotlib / plotly handles (re-exported for notebook convenience)
    "Axes3D",
    "BoundaryNorm",
    "LinearSegmentedColormap",
    "ListedColormap",
    "Polygon",
    "Rectangle",
    "TwoSlopeNorm",
    "apply_decorations",
    "ensure_axes",
    "go",
    "gridspec",
    "imshow_colorbar_caxdivider",
    "plt",
    # descriptive
    "plot_correlation_matrix",
    "plot_eigenvalue_spectrum",
    "plot_entropy_and_C",
    "plot_signed_balance",
    "plot_signed_laplacian_spectrum",
    "plot_weight_distribution",
    # filtering
    "plot_filtered_comparison",
    "plot_percolation_curve",
    # network
    "plot_network",
    "plot_node_metrics",
    "plot_signed_network",
    # lrg (per-PipelineResult + multiscale)
    "plot_dendrogram_with_psi",
    "plot_lrg_dendrogram",
    "plot_lrg_entropy",
    "plot_lrg_partition_network",
    "plot_lrg_psi",
    "plot_lrg_sankey",
    "plot_metastable_overlay",
    "plot_partition_flow",
    "plot_psi_curve",
    "plot_rmi_curve",
    "plot_specific_heat",
    "plot_specific_heat_overlay",
    "plot_tanglegram",
    # grids
    "plot_pipeline_summary",
    "plot_results_grid",
    # sankey
    "plot_sankey",
    "plot_sankey_matplotlib",
]
