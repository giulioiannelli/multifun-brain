"""End-to-end analysis pipeline for signed correlation matrices.

Provides a configurable, three-section pipeline that takes a raw
correlation matrix and produces structured results containing:

1. **Descriptive analysis** of the raw signed dense network (weight
   distribution, spectrum, precision matrix, signed Laplacian, metrics).
2. **Network filtering** via 2-3 methods that convert the signed matrix
   into unsigned tractable networks.
3. **Standard network analysis + LRG multiscale** on each filtered
   network (graph metrics, communities, diffusion clustering).
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .analysis.corrmatrix import (
    detect_dead_regions,
    hierarchical_partitions_from_corr,
    load_correlation_matrix,
    prepare_correlation_matrix,
)
from .analysis.descriptive import descriptive_report
from .analysis.filtering import apply_all_filters
from .analysis.netmetrics import network_summary_report

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "ResultsCollection",
    "discover_matrices",
    "load_results",
    "run_pipeline",
    "run_pipeline_batch",
    "run_pipeline_directory",
]

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Configuration for the three-section analysis pipeline.

    All parameters have sensible defaults.  Adjust *gamma* to match
    your data's aspect ratio (``n_regions / n_timepoints``).

    Attributes
    ----------
    gamma : float or None
        Aspect ratio p/n for RMT.  ``None`` disables MP comparison and
        MP-validated filtering.
    sigma : float
        Noise variance for MP density.
    precision_method : str
        Method for precision-matrix estimation (``'direct'``, ``'orie'``,
        ``'graphical_lasso'``).
    precision_alpha : float
        Regularisation for graphical-lasso.
    n_signed_modes : int
        Number of signed-Laplacian eigenmodes to keep.
    filter_methods : list of str
        Filtering methods to apply.
    filter_threshold : float or None
        Edge threshold for absolute / split methods.  ``None`` (default)
        auto-computes the percolation threshold at first-node detachment.
    filter_alpha : float
        Significance level for backbone extraction.
    tau_values : sequence of float or None
        Diffusion timescales for LRG.  Default: ``logspace(-2, 1, 6)``.
    normalized_laplacian : bool
        Whether to use the normalised Laplacian in LRG.
    run_lrg : bool
        Whether to run LRG multiscale analysis.
    run_standard_metrics : bool
        Whether to compute standard unsigned network metrics.
    run_community_detection : bool
        Whether to run Louvain community detection.
    run_rich_club : bool
        Whether to compute the rich-club curve (slow).
    seed : int or None
        Random seed for stochastic algorithms.
    """

    # Section 1: Descriptive analysis
    gamma: float | None = None
    sigma: float = 1.0
    precision_method: str = "direct"
    precision_alpha: float = 0.01
    n_signed_modes: int = 10

    # Section 2: Filtering
    filter_methods: list[str] = field(
        default_factory=lambda: ["absolute", "positive", "negative"]
    )
    filter_threshold: float | None = None
    filter_alpha: float = 0.05

    # Section 3: LRG multiscale
    tau_values: Sequence[float] | None = None
    normalized_laplacian: bool = True
    run_lrg: bool = True

    # Post-filter standard metrics
    run_standard_metrics: bool = True
    run_community_detection: bool = True
    run_rich_club: bool = False

    seed: int | None = None


# ──────────────────────────────────────────────────────────────────────
# Results container
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Container for all pipeline outputs.

    Attributes
    ----------
    config : PipelineConfig
        Configuration used for this run.
    label : str or None
        User-provided label for this result.
    n_regions_original : int
        Number of regions in the raw input matrix (before dropping dead regions).
    dropped_regions : np.ndarray
        Indices of dead regions that were dropped (empty array if none).
    corr_raw : np.ndarray
        The input correlation matrix.
    corr_prepared : np.ndarray
        Cleaned (symmetrised, clipped, diagonal zeroed) matrix, with dead
        regions removed.
    descriptive : dict
        Output of :func:`descriptive_report`.
    filtered_networks : dict
        ``{filter_name: {'graph': nx.Graph, 'nodes_removed': list}}``.
    network_analyses : dict
        ``{filter_name: network_summary_report()}``.
    lrg_results : dict
        ``{filter_name: list of partition dicts}``.
    """

    config: PipelineConfig
    label: str | None = None
    n_regions_original: int = 0
    dropped_regions: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    corr_raw: np.ndarray | None = None
    corr_prepared: np.ndarray | None = None
    descriptive: dict[str, Any] | None = None
    filtered_networks: dict[str, dict[str, Any]] = field(default_factory=dict)
    network_analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    lrg_results: dict[str, list[dict]] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise results to a plain dict (DataFrames become dicts)."""
        d: dict[str, Any] = {
            "label": self.label,
            "error": self.error,
            "n_regions_original": self.n_regions_original,
            "n_regions_after_cleanup": int(self.n_regions_original - len(self.dropped_regions)),
            "dropped_regions": self.dropped_regions.tolist(),
            "config": self.config.__dict__,
            "descriptive": _sanitise(self.descriptive),
            "filtered_networks": {
                name: {
                    "n_nodes": info["graph"].number_of_nodes(),
                    "n_edges": info["graph"].number_of_edges(),
                    "nodes_removed": info["nodes_removed"],
                    **(
                        {"percolation": {
                            k: v for k, v in info["percolation"].items()
                            if k != "max_per_node"
                        }}
                        if info.get("percolation") else {}
                    ),
                }
                for name, info in self.filtered_networks.items()
            },
            "network_analyses": {
                name: _sanitise(analysis)
                for name, analysis in self.network_analyses.items()
            },
            "lrg_results": {
                name: (
                    parts
                    if isinstance(parts, dict) and "error" in parts
                    else [
                        {"tau": p["tau"], "n_clusters": len(np.unique(p["partition"]))}
                        for p in parts
                    ]
                )
                for name, parts in self.lrg_results.items()
            },
        }
        return d

    def summary_table(self) -> pd.DataFrame:
        """One-row DataFrame of key metrics for easy concatenation across runs."""
        rows = []
        for fname, analysis in self.network_analyses.items():
            gm = analysis.get("global_metrics", {})
            comm = analysis.get("community", {})
            row = {
                "label": self.label,
                "n_regions_original": self.n_regions_original,
                "n_dropped": len(self.dropped_regions),
                "filter": fname,
                "n_nodes": gm.get("n_nodes"),
                "n_edges": gm.get("n_edges"),
                "density": gm.get("density"),
                "avg_clustering": gm.get("avg_clustering"),
                "avg_shortest_path": gm.get("avg_shortest_path"),
                "global_efficiency": gm.get("global_efficiency"),
                "assortativity": gm.get("assortativity"),
                "modularity": comm.get("modularity"),
                "n_communities": comm.get("n_communities"),
            }
            rows.append(row)
        return pd.DataFrame(rows)


def _sanitise(obj: Any) -> Any:
    """Recursively convert numpy arrays and DataFrames for JSON-like output."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# ──────────────────────────────────────────────────────────────────────
# Pipeline functions
# ──────────────────────────────────────────────────────────────────────


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
        :func:`~.analysis.corrmatrix.load_correlation_matrix`.
    config : PipelineConfig or None
        Pipeline configuration.  Uses defaults if *None*.
    label : str or None
        Optional label for the result (e.g., subject ID, band name).

    Returns
    -------
    PipelineResult
        Structured container with all analysis outputs.
    """
    if config is None:
        config = PipelineConfig()

    # ── Load ──────────────────────────────────────────────────────────
    if isinstance(corr, (str, Path)):
        log.info("Loading correlation matrix from %s", corr)
        label = label or Path(corr).stem
        corr_raw = load_correlation_matrix(corr)
    else:
        corr_raw = np.asarray(corr, dtype=float)

    # ── Detect dead regions before cleaning ───────────────────────────
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

    # ── Prepare (drops dead regions, symmetrises, clips) ─────────────
    corr_prepared = prepare_correlation_matrix(corr_raw)

    result = PipelineResult(
        config=config,
        label=label,
        n_regions_original=n_original,
        dropped_regions=dead,
        corr_raw=corr_raw,
        corr_prepared=corr_prepared,
    )

    # ── Guard: skip entirely if the prepared matrix is degenerate ─────
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

    # ── Section 1: Descriptive analysis ──────────────────────────────
    log.info("Section 1: Descriptive analysis")
    result.descriptive = descriptive_report(
        corr_prepared,
        gamma=config.gamma,
        sigma=config.sigma,
        precision_method=config.precision_method,
        precision_alpha=config.precision_alpha,
        n_signed_modes=config.n_signed_modes,
    )

    # ── Section 2: Filtering ─────────────────────────────────────────
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

    # ── Section 3: Analysis on each filtered network ─────────────────
    tau_values = (
        config.tau_values
        if config.tau_values is not None
        else np.logspace(-2, 1, 6)
    )

    for fname, fdata in result.filtered_networks.items():
        G = fdata["graph"]
        log.info("Section 3: Analysing filtered network '%s' (%d nodes, %d edges)",
                 fname, G.number_of_nodes(), G.number_of_edges())

        if G.number_of_nodes() == 0:
            log.warning("Filter '%s' produced an empty graph; skipping.", fname)
            continue

        # Standard network metrics
        if config.run_standard_metrics:
            result.network_analyses[fname] = network_summary_report(
                G,
                weight="weight",
                run_community_detection=config.run_community_detection,
                run_rich_club=config.run_rich_club,
                seed=config.seed,
            )

        # LRG multiscale
        if config.run_lrg:
            corr_for_lrg = nx.to_numpy_array(G)
            try:
                result.lrg_results[fname] = hierarchical_partitions_from_corr(
                    corr_for_lrg,
                    tau_values=tau_values,
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
        One result per input.
    """
    if labels is None:
        labels = [None] * len(inputs)  # type: ignore[list-item]
    return [
        run_pipeline(inp, config=config, label=lbl)
        for inp, lbl in zip(inputs, labels)
    ]


# ──────────────────────────────────────────────────────────────────────
# Directory discovery
# ──────────────────────────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS = {".npy", ".npz", ".csv", ".txt", ".pkl"}


def discover_matrices(
    directory: str | Path,
    extensions: set[str] | None = None,
    recursive: bool = True,
    pattern: str | None = None,
) -> list[Path]:
    """Find all correlation matrix files under *directory*.

    Parameters
    ----------
    directory : str or Path
        Root directory to search.
    extensions : set of str or None
        File extensions to include (default: ``.npy .npz .csv .txt .pkl``).
    recursive : bool
        If ``True`` (default), search subdirectories recursively.
    pattern : str or None
        If given, only include files whose name contains *pattern*
        (case-insensitive).  Useful for selecting e.g. only ``Bold``
        or ``Vaso`` matrices.

    Returns
    -------
    list of Path
        Sorted list of discovered file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    exts = extensions or _SUPPORTED_EXTENSIONS
    glob_method = directory.rglob if recursive else directory.glob
    files: list[Path] = []
    for ext in sorted(exts):
        files.extend(glob_method(f"*{ext}"))
    if pattern:
        pat = pattern.lower()
        files = [f for f in files if pat in f.name.lower()]
    return sorted(files)


def _label_from_path(path: Path, root: Path) -> str:
    """Derive a human-readable label from a file path relative to *root*."""
    rel = path.relative_to(root)
    # Use parent dirs + stem, joined by "/"
    parts = list(rel.parent.parts) + [rel.stem]
    return "/".join(parts)


def run_pipeline_directory(
    directory: str | Path,
    config: PipelineConfig | None = None,
    recursive: bool = True,
    pattern: str | None = None,
    extensions: set[str] | None = None,
) -> list[PipelineResult]:
    """Discover matrices in a directory tree and run the pipeline on each.

    This is the main convenience function for batch analysis.  Point it
    at a folder of correlation matrices (possibly nested in subfolders)
    and get back a list of ``PipelineResult`` objects.

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
        One result per discovered matrix file.

    Examples
    --------
    >>> from multifunbrain.pipeline import run_pipeline_directory, PipelineConfig
    >>> results = run_pipeline_directory(
    ...     "data/correlation_matrices",
    ...     config=PipelineConfig(gamma=0.19, filter_methods=["absolute", "positive"]),
    ... )
    >>> # Quick summary table across all matrices:
    >>> import pandas as pd
    >>> pd.concat([r.summary_table() for r in results], ignore_index=True)
    """
    directory = Path(directory)
    files = discover_matrices(
        directory,
        extensions=extensions,
        recursive=recursive,
        pattern=pattern,
    )
    log.info("Discovered %d matrices in %s", len(files), directory)
    labels = [_label_from_path(f, directory) for f in files]
    return run_pipeline_batch(files, config=config, labels=labels)


# ──────────────────────────────────────────────────────────────────────
# Results loading and access
# ──────────────────────────────────────────────────────────────────────


class ResultsCollection:
    """List-like container for :class:`PipelineResult` with label-based access.

    Supports integer indexing, iteration, ``len()``, and label-based
    lookup via ``results["my_label"]`` or ``results.labels``.

    Examples
    --------
    >>> results = load_results("pipeline_results/")
    >>> results[0]                    # first result
    >>> results["my_label"]           # by label
    >>> results.labels                # all labels
    >>> results.summary_table()       # combined summary across all results
    >>> for r in results:             # iterate
    ...     print(r.label)
    """

    def __init__(self, results: list[PipelineResult]) -> None:
        self._results = list(results)
        self._label_index: dict[str, int] = {}
        for i, r in enumerate(self._results):
            if r.label is not None:
                self._label_index[r.label] = i

    def __getitem__(self, key: int | str) -> PipelineResult:
        if isinstance(key, str):
            idx = self._label_index.get(key)
            if idx is None:
                raise KeyError(f"No result with label {key!r}")
            return self._results[idx]
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __repr__(self) -> str:
        return f"ResultsCollection({len(self)} results)"

    @property
    def labels(self) -> list[str | None]:
        """List of labels in order."""
        return [r.label for r in self._results]

    def filter(self, pattern: str) -> ResultsCollection:
        """Return a new collection keeping only results whose label contains *pattern*."""
        pat = pattern.lower()
        return ResultsCollection(
            [r for r in self._results if r.label and pat in r.label.lower()]
        )

    def summary_table(self) -> pd.DataFrame:
        """Concatenated summary table across all results."""
        frames = [r.summary_table() for r in self._results if r.network_analyses]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def load_results(path: str | Path) -> ResultsCollection:
    """Load pipeline results from a previous run.

    Parameters
    ----------
    path : str or Path
        Either the ``results.pkl`` file directly, or the output directory
        that contains it (e.g. ``"pipeline_results/"``).

    Returns
    -------
    ResultsCollection
        Results with label-based indexing, iteration, and summary table.

    Examples
    --------
    >>> from multifunbrain.pipeline import load_results
    >>> results = load_results("pipeline_results/")
    >>> results.labels
    ['mean_corr_HarvardOxford_48Parcels_kwfurN_Bold.ts.1D', ...]
    >>> r = results[0]
    >>> r.descriptive["weight_distribution"]["mean"]
    0.245
    """
    path = Path(path)
    if path.is_dir():
        path = path / "results.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    if isinstance(data, list):
        return ResultsCollection(data)
    return ResultsCollection([data])
