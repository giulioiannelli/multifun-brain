"""End-to-end multiscale LRG runner for a single correlation matrix.

Chain:

1. **MP denoise** the signed correlation matrix (gamma per acquisition).
2. **Backbone sparsification** on ``max(C̃, 0)`` (positive-only by default)
   via disparity or MP-validated.
3. **LRG on the backbone**: spectrum → ``S(τ)``, ``C(τ)`` → ``τ' = 1/λ_max``
   and ``τ*`` (specific-heat peak).
4. **Multiscale partitions**: for each ``τ`` on a log-grid spanning
   ``[τ', 3·τ*]``, compute the communication-distance dendrogram and pick
   the optimal cut via ``argmax Ψ(n; τ)``.
5. **Cross-τ metastability**: nodes whose cluster-membership set changes
   along the τ-trajectory.

Produces a :class:`MultiscaleResult` ready to be pickled and consumed by
the visual layer.

References
----------
Villegas, Gabrielli, Poggialini, Gili. *Phys. Rev. Research* 7, 013065 (2025).
Villegas, Gili, Caldarelli, Gabrielli. *Nat. Phys.* 19, 445 (2023).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster

from ..analysis.graphutils import compute_normalized_linkage
from ..analysis.lrg.distance import symmetrized_inverse_distance
from ..analysis.lrg.kernel import entropy, graph_laplacian_and_spectrum, rho_matrix
from ..analysis.lrg.metastable import metastable_nodes
from ..analysis.lrg.partitions import (
    compute_optimal_threshold,
    partition_stability_index,
)
from ..analysis.lrg.scales import characteristic_scales
from ..preprocessing.denoising import marchenko_pastur_denoise
from ..preprocessing.prepare import prepare_correlation_matrix
from ..processing.backbone import filter_validated

__all__ = ["MultiscaleResult", "run_multiscale_lrg"]

log = logging.getLogger(__name__)


@dataclass
class MultiscaleResult:
    """Container for one matrix's full multiscale LRG output."""

    label: str
    backbone_method: str
    backbone_alpha: float
    backbone_weights: str
    gamma: float | None
    sigma: float
    normalized_laplacian: bool

    n_nodes_original: int = 0
    nodes_alive: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    corr_mp_denoised: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    backbone_adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))

    time_grid_broad: np.ndarray = field(default_factory=lambda: np.zeros(0))
    entropy_1mS: np.ndarray = field(default_factory=lambda: np.zeros(0))
    specific_heat: np.ndarray = field(default_factory=lambda: np.zeros(0))

    tau_prime: float = float("nan")
    tau_star: float = float("nan")

    tau_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_tau: list[dict[str, Any]] = field(default_factory=list)

    metastable: dict[int, dict[str, Any]] = field(default_factory=dict)

    error: str | None = None


def run_multiscale_lrg(
    corr_raw: np.ndarray,
    *,
    gamma: float | None,
    backbone: str = "mp_validated",
    alpha: float = 0.05,
    backbone_weights: str = "positive",
    apply_mp_denoise: bool = True,
    sigma: float = 1.0,
    n_tau: int = 30,
    time_grid_steps: int = 600,
    log10_t1: float = -2.0,
    log10_t2: float = 5.0,
    normalized_laplacian: bool = True,
    label: str = "",
) -> MultiscaleResult:
    """Run the full MP → backbone → multiscale LRG chain on one matrix.

    Parameters
    ----------
    corr_raw : np.ndarray
        Signed correlation matrix (square, may contain NaN from dead ROIs).
    gamma : float or None
        Aspect ratio ``p / n`` for MP denoise. If ``None``, the input is
        only prepared (NaN-handled, symmetrised) without MP shrinkage —
        useful for benchmarking on data with unknown ``T``.
    backbone : str
        ``'disparity'`` | ``'lans'`` | ``'mp_validated'``.
    alpha : float
        Backbone significance level.
    backbone_weights : str
        ``'positive'`` (clip negatives to 0) or ``'abs'`` (keep
        anti-correlations as edge weight).
    sigma : float
        Noise variance estimate for MP denoise / validated filter.
    n_tau : int
        Number of log-spaced τ values for the per-τ dendrogram pass.
    time_grid_steps, log10_t1, log10_t2 : int, float, float
        Broad-grid parameters for ``S(τ)``, ``C(τ)``.
    normalized_laplacian : bool
        Use the symmetric normalised Laplacian if ``True``.
    label : str
        Identifier carried into the result (e.g. ``'co2/bpfBOLD'``).

    Returns
    -------
    MultiscaleResult
    """
    result = MultiscaleResult(
        label=label,
        backbone_method=backbone,
        backbone_alpha=alpha,
        backbone_weights=backbone_weights,
        gamma=gamma,
        sigma=sigma,
        normalized_laplacian=normalized_laplacian,
        n_nodes_original=int(np.asarray(corr_raw).shape[0]),
    )

    if apply_mp_denoise and gamma is not None:
        corr_clean = marchenko_pastur_denoise(corr_raw, gamma=gamma, sigma=sigma)
    else:
        corr_clean = prepare_correlation_matrix(corr_raw)
    result.corr_mp_denoised = corr_clean

    try:
        graph, _removed = filter_validated(
            corr_clean,
            method=backbone,
            alpha=alpha,
            gamma=gamma,
            sigma=sigma,
            weights=backbone_weights,
        )
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"backbone {backbone!r} failed: {exc!r}"
        log.warning("[%s] %s", label or "?", result.error)
        return result

    n_alive = graph.number_of_nodes()
    if n_alive < 4:
        result.error = (
            f"backbone collapsed to {n_alive} node(s); insufficient for LRG"
        )
        log.warning("[%s] %s", label or "?", result.error)
        return result

    nodes_alive = np.array(sorted(graph.nodes()), dtype=int)
    result.nodes_alive = nodes_alive

    adjacency = nx.to_numpy_array(graph, nodelist=nodes_alive, weight="weight")
    result.backbone_adjacency = adjacency

    L, spectrum = graph_laplacian_and_spectrum(
        graph, weight="weight", normalized=normalized_laplacian
    )
    result.eigenvalues = spectrum

    zero_tol = 1e-9
    if (np.abs(spectrum) > zero_tol).sum() < 2:
        result.error = (
            "Laplacian has fewer than two nonzero eigenvalues "
            f"(spectrum max={float(np.max(np.abs(spectrum))):.2e}); "
            "backbone too degenerate for LRG"
        )
        log.warning("[%s] %s", label or "?", result.error)
        return result

    one_minus_S, dS, _var_L, time_grid = entropy(
        spectrum, steps=time_grid_steps, t1=log10_t1, t2=log10_t2
    )
    result.entropy_1mS = one_minus_S
    result.time_grid_broad = time_grid
    specific_heat = np.concatenate([dS, [dS[-1]]]) if dS.size + 1 == time_grid.size else dS
    if specific_heat.size != time_grid.size:
        specific_heat = np.pad(dS, (0, time_grid.size - dS.size), constant_values=np.nan)
    result.specific_heat = specific_heat

    try:
        scales = characteristic_scales(
            spectrum, time_grid, specific_heat, n_tau=n_tau
        )
    except ValueError as exc:
        result.error = f"characteristic_scales failed: {exc!r}"
        log.warning("[%s] %s", label or "?", result.error)
        return result
    result.tau_prime = float(scales["tau_prime"])
    result.tau_star = float(scales["tau_star"])
    tau_grid = np.asarray(scales["tau_grid_default"], dtype=float)
    result.tau_grid = tau_grid

    per_tau: list[dict[str, Any]] = []
    for tau in tau_grid:
        dists = symmetrized_inverse_distance(tau, lambda t: rho_matrix(t, L))
        linkage_matrix, _labels, tmax = compute_normalized_linkage(
            dists, graph, labelList="numbers"
        )
        flat_threshold, _opt_threshold, _stab, opt_idx = compute_optimal_threshold(
            linkage_matrix
        )
        psi_n = partition_stability_index(linkage_matrix)
        partition = fcluster(linkage_matrix, flat_threshold, criterion="distance")
        per_tau.append({
            "tau": float(tau),
            "partition": np.asarray(partition, dtype=int),
            "linkage": linkage_matrix,
            "psi_n": psi_n,
            "optimal_branch_idx": int(opt_idx),
            "n_clusters": int(partition.max()) if partition.size else 0,
            "flat_threshold": float(flat_threshold),
            "tmax": float(tmax),
        })
    result.per_tau = per_tau

    result.metastable = metastable_nodes(per_tau)
    return result
