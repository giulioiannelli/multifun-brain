"""Private helpers shared across the ``plotlib`` section files.

These are not part of the public API. They live here (not in ``base.py``)
because they're pipeline-specific — they assume the
:class:`~multifunbrain.pipeline.PipelineResult` /
:class:`~multifunbrain.pipeline.multiscale.MultiscaleResult` shapes — and
exposing them would tempt callers to write ad-hoc plot code instead of
adding helpers to this package.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

__all__ = [
    "_get_section",
    "_nearest_tau_index",
    "_network_layout",
    "_resolve_filter",
    "_signed_laplacian_embedding",
]


def _get_section(result: Any, key: str) -> dict[str, Any]:
    """Extract a descriptive sub-dict from a PipelineResult or raw dict."""
    if isinstance(result, dict):
        # Already the sub-dict
        if key in result:
            return result[key]
        return result
    # PipelineResult
    desc = getattr(result, "descriptive", None)
    if desc is None:
        raise ValueError("Result has no descriptive analysis. Run the pipeline first.")
    if key in desc:
        return desc[key]
    raise KeyError(
        f"Key {key!r} not found in descriptive results. "
        f"Available: {list(desc.keys())}"
    )


def _resolve_filter(result: Any, filter_name: str | None) -> str:
    """Pick a filter name, defaulting to the first available."""
    if filter_name is not None:
        return filter_name
    if result.filtered_networks:
        return next(iter(result.filtered_networks))
    raise ValueError("No filtered networks available in this result.")


def _signed_laplacian_embedding(corr: np.ndarray) -> dict:
    """2-D spectral embedding from the signed Laplacian.

    The two eigenvectors of :math:`L_s = |D| - A` associated with the
    smallest eigenvalues capture the dominant bipartition of the signed
    network, placing nodes in the same "balanced community" close
    together.
    """
    C = corr.copy()
    np.fill_diagonal(C, 0.0)
    abs_C = np.abs(C)
    D = np.diag(abs_C.sum(axis=1))
    L_s = D - C

    eigenvalues, eigenvectors = np.linalg.eigh(L_s)
    # Smallest eigenvalues capture the frustrated bipartition.
    idx = np.argsort(eigenvalues)
    coords = eigenvectors[:, idx[:2]]

    # Normalise to unit square for consistent rendering.
    for dim in range(2):
        r = coords[:, dim].max() - coords[:, dim].min()
        if r > 0:
            coords[:, dim] = (coords[:, dim] - coords[:, dim].min()) / r

    return {i: coords[i] for i in range(len(coords))}


def _network_layout(G: nx.Graph, signed: bool = False) -> dict:
    """Compute a layout that reveals modular structure.

    For unsigned networks (all positive weights), uses Kamada-Kawai with
    distance = 1/weight so strongly connected nodes are placed close.
    For signed networks, uses spring layout where positive weights
    attract and negative weights repel.
    """
    if G.number_of_nodes() == 0:
        return {}
    if signed:
        return nx.spring_layout(G, weight="weight", seed=42, iterations=100)
    # Kamada-Kawai needs a distance matrix; invert weights so strong = close.
    try:
        return nx.kamada_kawai_layout(G, weight="weight")
    except (ValueError, nx.NetworkXError):
        return nx.spring_layout(G, weight="weight", seed=42, iterations=100)


def _nearest_tau_index(tau_grid: np.ndarray, tau_value: float) -> int:
    """Index of the τ in ``tau_grid`` closest to ``tau_value``."""
    return int(np.argmin(np.abs(np.asarray(tau_grid) - tau_value)))
