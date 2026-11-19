"""Laplacian Renormalisation Group (LRG) machinery.

Canonical home for the diffusion-kernel + hierarchical-clustering tools.
Imports of the legacy module ``multifunbrain.analysis.lrglib`` are
re-routed here.

Submodules:

- :mod:`multifunbrain.analysis.lrg.kernel`: Laplacian, ``rho_matrix``, entropy.
- :mod:`multifunbrain.analysis.lrg.distance`: diffusion-based distance.
- :mod:`multifunbrain.analysis.lrg.partitions`: hierarchical partitions and
  switching-node helpers.
"""

from __future__ import annotations

from .compare import (
    HierarchyComparison,
    bakers_gamma,
    compare_hierarchies,
    cophenetic_correlation,
    null_shift_distribution,
    reduced_mutual_information,
    specific_heat_comparison,
)
from .distance import symmetrized_inverse_distance
from .kernel import entropy, graph_laplacian_and_spectrum, rho_matrix
from .layout import diffusion_distance_matrix, diffusion_distance_mds
from .metastable import metastable_nodes
from .partitions import (
    compute_optimal_threshold,
    get_moved_nodes,
    get_moved_nodes_interval,
    hierarchical_partitions_from_corr,
    identify_switching_nodes,
    linkage_at_tau_min,
    local_partition_stability_index,
    partition_flow_table,
    partition_stability_index,
)
from .scales import characteristic_scales

__all__ = [
    "HierarchyComparison",
    "bakers_gamma",
    "characteristic_scales",
    "compare_hierarchies",
    "compute_optimal_threshold",
    "cophenetic_correlation",
    "diffusion_distance_matrix",
    "diffusion_distance_mds",
    "entropy",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    "graph_laplacian_and_spectrum",
    "hierarchical_partitions_from_corr",
    "identify_switching_nodes",
    "linkage_at_tau_min",
    "local_partition_stability_index",
    "metastable_nodes",
    "null_shift_distribution",
    "partition_flow_table",
    "partition_stability_index",
    "reduced_mutual_information",
    "rho_matrix",
    "specific_heat_comparison",
    "symmetrized_inverse_distance",
]
