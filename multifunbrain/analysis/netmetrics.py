"""Back-compat shim for :mod:`multifunbrain.analysis.network`.

The standard-network-metrics functions used to live here as a flat
module. Each is now in its own scoped submodule of
:mod:`multifunbrain.analysis.network`. This module re-exports them so
existing imports keep working.

New code should import from :mod:`multifunbrain.analysis.network` directly.
"""

from __future__ import annotations

from .network import (
    compute_global_metrics,
    compute_node_metrics,
    compute_rich_club_curve,
    degree_distribution_analysis,
    detect_communities_louvain,
    detect_communities_spectral,
    network_summary_report,
)

__all__ = [
    "compute_global_metrics",
    "compute_node_metrics",
    "compute_rich_club_curve",
    "degree_distribution_analysis",
    "detect_communities_louvain",
    "detect_communities_spectral",
    "network_summary_report",
]
