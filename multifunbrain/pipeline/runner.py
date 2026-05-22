"""Pipeline orchestrators: run, batch, and directory runners."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import numpy as np

from ..analysis.descriptive.report import descriptive_report
from ..analysis.lrg.kernel import entropy as lrg_entropy
from ..analysis.lrg.kernel import graph_laplacian_and_spectrum
from ..analysis.lrg.partitions import hierarchical_partitions_from_corr
from ..analysis.lrg.scales import characteristic_scales
from ..analysis.network.report import network_summary_report
from ..io.corrmatrix import load_correlation_matrix
from ..preprocessing.dead_regions import detect_dead_regions
from ..preprocessing.prepare import prepare_correlation_matrix
from ..processing.filtering import apply_all_filters
from .config import PipelineConfig
from .discovery import discover_matrices, label_from_path
from .result import PipelineResult

__all__ = ["run_pipeline", "run_pipeline_batch", "run_pipeline_directory"]

log = logging.getLogger(__name__)


def run_pipeline(
    corr: np.ndarray | str | Path,
    config: PipelineConfig | None = None,
    label: str | None = None,
) -> PipelineResult:
    """Run the full three-section analysis pipeline on a correlation matrix.

    Parameters
    ----------
    corr : np.ndarray or str or Path
        A square correlation matrix, or a path to a file loadable by
        :func:`~multifunbrain.io.corrmatrix.load_correlation_matrix`.
    config : PipelineConfig or None
        Pipeline configuration. Uses defaults if *None*.
    label : str or None
        Optional label for the result (e.g., subject ID, band name).

    Returns
    -------
    PipelineResult
        Structured container with all analysis outputs.
    """
    if config is None:
        config = PipelineConfig()

    if isinstance(corr, (str, Path)):
        log.info("Loading correlation matrix from %s", corr)
        label = label or Path(corr).stem
        corr_raw = load_correlation_matrix(corr)
    else:
        corr_raw = np.asarray(corr, dtype=float)

    n_original = corr_raw.shape[0]
    dead = detect_dead_regions(corr_raw)
    if len(dead) > 0:
        log.info(
            "[%s] %d dead region(s) detected (indices: %s); "
            "matrix will be reduced from %d to %d regions.",
            label or "?",
            len(dead),
            dead.tolist(),
            n_original,
            n_original - len(dead),
        )

    corr_prepared = prepare_correlation_matrix(corr_raw)

    result = PipelineResult(
        config=config,
        label=label,
        n_regions_original=n_original,
        dropped_regions=dead,
        corr_raw=corr_raw,
        corr_prepared=corr_prepared,
    )

    n_alive = corr_prepared.shape[0]
    if n_alive < 2:
        msg = (
            f"Only {n_alive} region(s) survived after removing "
            f"{len(dead)}/{n_original} dead regions; "
            f"skipping analysis"
        )
        log.warning("[%s] %s", label or "?", msg)
        result.error = msg
        return result

    log.info("Section 1: Descriptive analysis")
    result.descriptive = descriptive_report(
        corr_prepared,
        gamma=config.gamma,
        sigma=config.sigma,
        precision_method=config.precision_method,
        precision_alpha=config.precision_alpha,
        n_signed_modes=config.n_signed_modes,
    )

    log.info("Section 2: Network filtering (%s)", config.filter_methods)
    result.filtered_networks = apply_all_filters(
        corr_prepared,
        methods=config.filter_methods,
        threshold=config.filter_threshold,
        alpha=config.filter_alpha,
        gamma=config.gamma,
        sigma=config.sigma,
        precision_method=config.precision_method,
        precision_alpha=config.precision_alpha,
    )

    for fname, fdata in result.filtered_networks.items():
        G = fdata["graph"]
        log.info(
            "Section 3: Analysing filtered network '%s' (%d nodes, %d edges)",
            fname,
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        if G.number_of_nodes() == 0:
            log.warning("Filter '%s' produced an empty graph; skipping.", fname)
            continue

        if config.run_standard_metrics:
            result.network_analyses[fname] = network_summary_report(
                G,
                weight="weight",
                run_community_detection=config.run_community_detection,
                run_rich_club=config.run_rich_club,
                seed=config.seed,
            )

        if config.run_lrg:
            # Tau grid: explicit override > per-filter adaptive grid from the
            # spectrum (tau_min = 1/lambda_max, tau_max ~ 3*tau_star where
            # C(tau) peaks). Falls back to the generic logspace(-2,1,6) only
            # if the adaptive grid cannot be computed.
            if config.tau_values is not None:
                tau_values_used = np.asarray(config.tau_values, dtype=float)
            else:
                try:
                    _, evals_f = graph_laplacian_and_spectrum(
                        G, weight="weight",
                        normalized=config.normalized_laplacian,
                    )
                    _, C_f, _, t_f = lrg_entropy(evals_f)
                    scales = characteristic_scales(evals_f, t_f[:-1], C_f)
                    tau_values_used = scales["tau_grid_default"]
                except Exception as exc:
                    log.warning(
                        "Adaptive tau grid failed for '%s' (%s); "
                        "falling back to logspace(-2,1,6).", fname, exc,
                    )
                    tau_values_used = np.logspace(-2, 1, 6)

            corr_for_lrg = nx.to_numpy_array(G)
            try:
                result.lrg_results[fname] = hierarchical_partitions_from_corr(
                    corr_for_lrg,
                    tau_values=tau_values_used,
                    normalized_laplacian=config.normalized_laplacian,
                )
            except Exception as exc:
                log.warning(
                    "LRG analysis failed for filter '%s': %s", fname, exc
                )
                result.lrg_results[fname] = {
                    "error": str(exc),
                    "filter": fname,
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": G.number_of_edges(),
                }

    return result


def run_pipeline_batch(
    inputs: Sequence[np.ndarray | str | Path],
    config: PipelineConfig | None = None,
    labels: Sequence[str] | None = None,
) -> list[PipelineResult]:
    """Run the pipeline on multiple correlation matrices.

    Parameters
    ----------
    inputs : sequence of (np.ndarray or str or Path)
        Correlation matrices or paths.
    config : PipelineConfig or None
        Shared configuration for all runs.
    labels : sequence of str or None
        Labels for each input.

    Returns
    -------
    list of PipelineResult
    """
    if labels is None:
        labels = [None] * len(inputs)  # type: ignore[list-item]
    return [
        run_pipeline(inp, config=config, label=lbl)
        for inp, lbl in zip(inputs, labels)
    ]


def run_pipeline_directory(
    directory: str | Path,
    config: PipelineConfig | None = None,
    recursive: bool = True,
    pattern: str | None = None,
    extensions: set[str] | None = None,
) -> list[PipelineResult]:
    """Discover matrices in a directory tree and run the pipeline on each.

    Parameters
    ----------
    directory : str or Path
        Root directory to search.
    config : PipelineConfig or None
        Pipeline configuration (shared across all matrices).
    recursive : bool
        Search subdirectories (default ``True``).
    pattern : str or None
        Filename filter (case-insensitive substring match).
    extensions : set of str or None
        File extensions to include.

    Returns
    -------
    list of PipelineResult
    """
    directory = Path(directory)
    files = discover_matrices(
        directory,
        extensions=extensions,
        recursive=recursive,
        pattern=pattern,
    )
    log.info("Discovered %d matrices in %s", len(files), directory)
    labels = [label_from_path(f, directory) for f in files]
    return run_pipeline_batch(files, config=config, labels=labels)
