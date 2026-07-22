"""Serializers: turn ``PipelineResult`` / ``MultiscaleResult`` into JSON plot specs.

One module per pipeline section. ``PLOT_KINDS`` maps a plot ``kind`` string to a
``(result, **params) -> dict`` function so routes stay declarative and the
frontend can discover available kinds. ``params`` carries the shared query
options (``filter_name``, ``tau_index``, ``edge_quantile``); each serializer
takes what it needs and ignores the rest via ``**_``.
"""

from __future__ import annotations

from collections.abc import Callable

from . import descriptive, lrg, network

PLOT_KINDS: dict[str, Callable] = {
    # descriptive (Section 1)
    "region_names": descriptive.region_names_spec,
    "heatmap": descriptive.heatmap_spec,
    "partial_correlation": descriptive.partial_correlation_spec,
    "precision": descriptive.precision_spec,
    "weights": descriptive.weights_spec,
    "spectrum": descriptive.spectrum_spec,
    "signed_laplacian": descriptive.signed_laplacian_spec,
    "signed_balance": descriptive.signed_balance_spec,
    # standard network (Section 3), per filter
    "global_metrics": network.global_metrics_spec,
    "degree_distribution": network.degree_distribution_spec,
    "node_metrics": network.node_metrics_spec,
    "network": network.network_spec,
    # LRG multiscale (Section 3), per filter
    "tau_grid": lrg.tau_grid_spec,
    "dendrogram": lrg.dendrogram_spec,
    "specific_heat": lrg.specific_heat_spec,
    "psi": lrg.psi_spec,
    "partition_flow": lrg.partition_flow_spec,
    "sankey": lrg.sankey_spec,
    "lrg_network": lrg.lrg_network_spec,
}
