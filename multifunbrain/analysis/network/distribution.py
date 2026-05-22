"""Degree distribution and rich-club analysis."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

__all__ = ["degree_distribution_analysis", "compute_rich_club_curve"]


def degree_distribution_analysis(
    G: nx.Graph,
    weight: str | None = None,
    n_bins: int = 30,
) -> dict[str, Any]:
    """Analyse the degree (or strength) distribution of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str or None
        If set, analyses strength (weighted degree) instead.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Distribution statistics and histogram.
    """
    from scipy.stats import kurtosis, skew

    if weight:
        values = np.array([d for _, d in G.degree(weight=weight)])
    else:
        values = np.array([d for _, d in G.degree()])

    hist_counts, hist_edges = np.histogram(values, bins=n_bins)

    return {
        "values": values,
        "histogram": (hist_counts, hist_edges),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skewness": float(skew(values)) if len(values) > 1 else 0.0,
        "kurtosis": float(kurtosis(values)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
    }


def compute_rich_club_curve(
    G: nx.Graph,
    normalized: bool = True,
    n_rand: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute the rich-club coefficient curve.

    Parameters
    ----------
    G : nx.Graph
        Input graph (unweighted for standard rich-club).
    normalized : bool
        If True, normalise against degree-preserving random graphs.
    n_rand : int
        Number of random realisations for normalisation.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        ``'k'`` (degree thresholds), ``'phi'`` (raw rich-club),
        and (if *normalized*) ``'phi_rand'`` (mean random) and
        ``'phi_norm'`` (normalised coefficients).
    """
    G_uw = nx.Graph()
    G_uw.add_nodes_from(G.nodes())
    G_uw.add_edges_from(G.edges())

    rc = nx.rich_club_coefficient(G_uw, normalized=False)
    if not rc:
        return {"k": np.array([]), "phi": np.array([])}

    ks = np.array(sorted(rc.keys()))
    phi = np.array([rc[k] for k in ks])

    result: dict[str, Any] = {"k": ks, "phi": phi}

    if normalized and len(ks) > 0:
        rng = np.random.default_rng(seed)
        phi_rand_sum = np.zeros_like(phi, dtype=float)

        for _ in range(n_rand):
            G_rand = _degree_preserving_randomisation(G_uw, rng)
            rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
            for idx, k in enumerate(ks):
                phi_rand_sum[idx] += rc_rand.get(k, 0.0)

        phi_rand = phi_rand_sum / n_rand
        phi_norm = np.divide(
            phi, phi_rand, out=np.ones_like(phi, dtype=float), where=phi_rand > 0
        )
        result["phi_rand"] = phi_rand
        result["phi_norm"] = phi_norm

    return result


def _degree_preserving_randomisation(
    G: nx.Graph,
    rng: np.random.Generator,
    n_swaps: int | None = None,
) -> nx.Graph:
    """Degree-preserving edge randomisation via double-edge swaps."""
    G_rand = G.copy()
    if n_swaps is None:
        n_swaps = G.number_of_edges() * 10
    try:
        nx.double_edge_swap(
            G_rand,
            nswap=n_swaps,
            max_tries=n_swaps * 10,
            seed=int(rng.integers(2**31)),
        )
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass  # best-effort
    return G_rand
