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
    bakers_gamma,
    cophenetic_correlation,
    reduced_mutual_information,
    specific_heat_comparison,
)
from .distance import symmetrized_inverse_distance
from .kernel import entropy, graph_laplacian_and_spectrum, rho_matrix
from .metastable import metastable_nodes
from .partitions import (
    compute_optimal_threshold,
    get_moved_nodes,
    get_moved_nodes_interval,
    hierarchical_partitions_from_corr,
    identify_switching_nodes,
    local_partition_stability_index,
    partition_flow_table,
    partition_stability_index,
)
from .scales import characteristic_scales

__all__ = [
    "bakers_gamma",
    "characteristic_scales",
    "compute_optimal_threshold",
    "cophenetic_correlation",
    "entropy",
    "get_moved_nodes",
    "get_moved_nodes_interval",
    "graph_laplacian_and_spectrum",
    "hierarchical_partitions_from_corr",
    "identify_switching_nodes",
    "local_partition_stability_index",
    "metastable_nodes",
    "partition_flow_table",
    "partition_stability_index",
    "reduced_mutual_information",
    "rho_matrix",
    "specific_heat_comparison",
    "symmetrized_inverse_distance",
]
