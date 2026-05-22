"""Standard network metrics for unsigned weighted graphs.

Submodules:

- :mod:`multifunbrain.analysis.network.global_metrics`: graph-level metrics.
- :mod:`multifunbrain.analysis.network.node_metrics`: per-node centralities.
- :mod:`multifunbrain.analysis.network.community`: Louvain & spectral clustering.
- :mod:`multifunbrain.analysis.network.distribution`: degree distribution &
  rich-club.
- :mod:`multifunbrain.analysis.network.report`: aggregate ``network_summary_report``.
"""

from __future__ import annotations

from .community import detect_communities_louvain, detect_communities_spectral
from .distribution import compute_rich_club_curve, degree_distribution_analysis
from .global_metrics import compute_global_metrics
from .node_metrics import compute_node_metrics
from .report import network_summary_report

__all__ = [
    "compute_global_metrics",
    "compute_node_metrics",
    "compute_rich_club_curve",
    "degree_distribution_analysis",
    "detect_communities_louvain",
    "detect_communities_spectral",
    "network_summary_report",
]
