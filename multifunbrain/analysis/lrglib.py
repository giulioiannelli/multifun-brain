"""Back-compat shim for :mod:`multifunbrain.analysis.lrg`.

The LRG kernel, distance, and partition routines used to live here as a
flat module; they now have a single canonical home in the
:mod:`multifunbrain.analysis.lrg` subpackage. This module re-exports
them so existing imports keep working.

New code should import from :mod:`multifunbrain.analysis.lrg` directly.
"""

from __future__ import annotations

from .lrg import (
    compute_optimal_threshold,
    entropy,
    get_moved_nodes,
    get_moved_nodes_interval,
    graph_laplacian_and_spectrum,
    hierarchical_partitions_from_corr,
    identify_switching_nodes,
    rho_matrix,
    symmetrized_inverse_distance,
)

__all__ = [
    "compute_optimal_threshold",
    "entropy",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    "graph_laplacian_and_spectrum",
    "hierarchical_partitions_from_corr",
    "identify_switching_nodes",
    "rho_matrix",
    "symmetrized_inverse_distance",
]
