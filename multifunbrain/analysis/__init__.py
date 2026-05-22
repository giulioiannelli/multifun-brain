"""Analysis routines for brain graphs.

This package aggregates the analysis-side public API. Symbols defined in
sibling packages (``io``, ``preprocessing``, ``processing``) are
re-exported here for ergonomic notebook use; their canonical home is
the original module.
"""

from __future__ import annotations

# Cross-package re-exports (canonical homes elsewhere)
from ..io.corrmatrix import load_correlation_matrix
from ..preprocessing.dead_regions import detect_dead_regions
from ..preprocessing.denoising import marchenko_pastur_denoise
from ..preprocessing.prepare import prepare_correlation_matrix
from ..processing.backbone import filter_validated
from ..processing.filtering import (
    apply_all_filters,
    filter_absolute_threshold,
    filter_partial_correlation,
    filter_split_sign,
)

# Local subpackages (canonical homes inside multifunbrain.analysis)
from .corrnet import (
    compute_correlation_matrix,
    marchenko_pastur,
    marchenko_pastur_density,
)
from .descriptive import (
    compute_precision_matrix,
    correlation_spectrum_analysis,
    descriptive_report,
    signed_laplacian_analysis,
    signed_laplacian_and_spectrum,
    signed_network_metrics,
    weight_distribution_analysis,
)
from .graphutils import (
    build_correlation_network,
    compute_normalized_linkage,
    compute_optimal_threshold_std,
    compute_threshold_stats,
    compute_threshold_stats_fast,
    find_threshold_jumps,
    get_giant_component,
    get_giant_component_leftoff,
    select_threshold_elbow,
    select_threshold_fraction,
    select_threshold_plateau,
)
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
from .network import (
    compute_global_metrics,
    compute_node_metrics,
    compute_rich_club_curve,
    degree_distribution_analysis,
    detect_communities_louvain,
    detect_communities_spectral,
    network_summary_report,
)
from .partition import adjusted_rand_index, compare_partition_sets

__all__ = [
    # corrnet
    "compute_correlation_matrix",
    "marchenko_pastur",
    "marchenko_pastur_density",
    # graphutils
    "build_correlation_network",
    "compute_normalized_linkage",
    "compute_optimal_threshold_std",
    "compute_threshold_stats",
    "compute_threshold_stats_fast",
    "find_threshold_jumps",
    "get_giant_component",
    "get_giant_component_leftoff",
    "select_threshold_elbow",
    "select_threshold_fraction",
    "select_threshold_plateau",
    # lrg
    "compute_optimal_threshold",
    "entropy",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    "graph_laplacian_and_spectrum",
    "hierarchical_partitions_from_corr",
    "identify_switching_nodes",
    "rho_matrix",
    "symmetrized_inverse_distance",
    # cross-package re-exports (io / preprocessing)
    "detect_dead_regions",
    "load_correlation_matrix",
    "marchenko_pastur_denoise",
    "prepare_correlation_matrix",
    # partition
    "adjusted_rand_index",
    "compare_partition_sets",
    # descriptive
    "compute_precision_matrix",
    "correlation_spectrum_analysis",
    "descriptive_report",
    "signed_laplacian_analysis",
    "signed_laplacian_and_spectrum",
    "signed_network_metrics",
    "weight_distribution_analysis",
    # processing filtering
    "apply_all_filters",
    "filter_absolute_threshold",
    "filter_partial_correlation",
    "filter_split_sign",
    "filter_validated",
    # network (standard unsigned metrics)
    "compute_global_metrics",
    "compute_node_metrics",
    "compute_rich_club_curve",
    "degree_distribution_analysis",
    "detect_communities_louvain",
    "detect_communities_spectral",
    "network_summary_report",
]
