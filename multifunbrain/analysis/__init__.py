"""Analysis routines for brain graphs."""

from .corrmatrix import (
    adjusted_rand_index,
    compare_partition_sets,
    detect_dead_regions,
    hierarchical_partitions_from_corr,
    load_correlation_matrix,
    marchenko_pastur_denoise,
    prepare_correlation_matrix,
)
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
from .filtering import (
    apply_all_filters,
    filter_absolute_threshold,
    filter_partial_correlation,
    filter_split_sign,
    filter_validated,
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
from .lrglib import (
    compute_optimal_threshold,
    entropy,
    get_moved_nodes,
    get_moved_nodes_interval,
    graph_laplacian_and_spectrum,
    identify_switching_nodes,
    rho_matrix,
    symmetrized_inverse_distance,
)
from .netmetrics import (
    compute_global_metrics,
    compute_node_metrics,
    compute_rich_club_curve,
    degree_distribution_analysis,
    detect_communities_louvain,
    detect_communities_spectral,
    network_summary_report,
)

__all__ = [
    # corrnet
    "compute_correlation_matrix",
    "marchenko_pastur",
    "marchenko_pastur_density",
    # graphutils
    "get_giant_component",
    "get_giant_component_leftoff",
    "build_correlation_network",
    "compute_threshold_stats",
    "compute_threshold_stats_fast",
    "find_threshold_jumps",
    "select_threshold_elbow",
    "select_threshold_fraction",
    "select_threshold_plateau",
    "compute_normalized_linkage",
    "compute_optimal_threshold_std",
    # lrglib
    "graph_laplacian_and_spectrum",
    "rho_matrix",
    "entropy",
    "symmetrized_inverse_distance",
    "compute_optimal_threshold",
    "identify_switching_nodes",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    # corrmatrix
    "detect_dead_regions",
    "load_correlation_matrix",
    "prepare_correlation_matrix",
    "marchenko_pastur_denoise",
    "hierarchical_partitions_from_corr",
    "adjusted_rand_index",
    "compare_partition_sets",
    # descriptive
    "weight_distribution_analysis",
    "correlation_spectrum_analysis",
    "compute_precision_matrix",
    "signed_laplacian_and_spectrum",
    "signed_laplacian_analysis",
    "signed_network_metrics",
    "descriptive_report",
    # filtering
    "filter_absolute_threshold",
    "filter_split_sign",
    "filter_validated",
    "filter_partial_correlation",
    "apply_all_filters",
    # netmetrics
    "compute_global_metrics",
    "compute_node_metrics",
    "detect_communities_louvain",
    "detect_communities_spectral",
    "degree_distribution_analysis",
    "compute_rich_club_curve",
    "network_summary_report",
]
