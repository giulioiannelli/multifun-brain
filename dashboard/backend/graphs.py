"""Shared graph derivation for the result tabs.

One canonical place that turns a ``PipelineResult`` + a *sparsify* selection into
the graph to analyse, so the Network, LRG and Brain-3D tabs all thin the network
the same way. ``sparsify`` re-derives a backbone from the stored signed
correlation (``corr_prepared``) with the pipeline's own methods; ``None`` /
``"filter"`` keeps the pre-computed filtered network unchanged. Falls back to the
stored network whenever the signed matrix is missing (older bundles) or the
backbone collapses, so a plot never errors on an ad-hoc sparsification.

Node ids of every returned graph index into ``surviving_labels(result)`` /
``surviving_coords(result)`` exactly as the stored filtered network does, so hover
names and 3-D coordinates stay aligned (the node-identity invariant).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# Sparsify ids the frontend may send. ``mp_validated`` is only offered when the
# result carries a Marchenko-Pastur aspect ratio (gamma).
SPARSIFY_METHODS = ("filter", "percolation", "disparity", "lans", "mp_validated", "threshold")


def result_gamma(result) -> float | None:
    """Marchenko-Pastur aspect ratio for this result, or ``None`` if unavailable."""
    gamma = getattr(getattr(result, "config", None), "gamma", None)
    try:
        return float(gamma) if gamma is not None else None
    except (TypeError, ValueError):
        return None


def derive_graph(
    result,
    fname: str,
    sparsify: str | None,
    alpha: float = 0.05,
    threshold: float = 0.3,
) -> tuple[nx.Graph, str]:
    """Return ``(graph, effective_sparsify)`` for *fname* under *sparsify*.

    ``sparsify`` ∈ :data:`SPARSIFY_METHODS`. Anything falsy / ``"filter"`` returns
    the stored filtered network. ``mp_validated`` needs ``result.config.gamma``.
    """
    stored: nx.Graph = result.filtered_networks[fname]["graph"]
    if sparsify in (None, "", "filter"):
        return stored, "filter"
    corr = getattr(result, "corr_prepared", None)
    if corr is None:
        return stored, "filter"
    corr = np.asarray(corr, dtype=float)
    try:
        if sparsify == "percolation":
            from multifunbrain.processing.backbone import filter_validated

            g, _ = filter_validated(corr, method="percolation", weights="abs")
        elif sparsify == "disparity":
            from multifunbrain.processing.backbone import filter_validated

            g, _ = filter_validated(corr, method="disparity", alpha=alpha, weights="positive")
        elif sparsify == "lans":
            from multifunbrain.processing.backbone import filter_validated

            g, _ = filter_validated(corr, method="lans", alpha=alpha, weights="positive")
        elif sparsify == "mp_validated":
            gamma = result_gamma(result)
            if gamma is None:
                return stored, "filter"
            from multifunbrain.processing.backbone import filter_validated

            # corr_prepared has a zeroed diagonal (C − I); the MP edge λ+ assumes a
            # unit-diagonal C, so restore it before eigen-thresholding.
            unit = corr.copy()
            np.fill_diagonal(unit, 1.0)
            g, _ = filter_validated(
                unit, method="mp_validated", gamma=gamma, weights="positive"
            )
        elif sparsify == "threshold":
            from multifunbrain.processing.filtering import filter_absolute_threshold

            g, _ = filter_absolute_threshold(corr, threshold=threshold)
        else:
            return stored, "filter"
    except (ValueError, np.linalg.LinAlgError):
        return stored, "filter"
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return stored, "filter"
    return g, sparsify
