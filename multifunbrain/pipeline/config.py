"""Pipeline configuration dataclass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["PipelineConfig"]


@dataclass
class PipelineConfig:
    """Configuration for the three-section analysis pipeline.

    All parameters have sensible defaults. Adjust *gamma* to match your
    data's aspect ratio (``n_regions / n_timepoints``).

    Attributes
    ----------
    gamma : float or None
        Aspect ratio p/n for RMT. ``None`` disables MP comparison and
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
        Edge threshold for absolute / split methods. ``None`` (default)
        auto-computes the percolation threshold at first-node detachment.
    filter_alpha : float
        Significance level for backbone extraction.
    tau_values : sequence of float or None
        Diffusion timescales for LRG. Default: ``logspace(-2, 1, 6)``.
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
        default_factory=lambda: ["absolute", "partial_correlation"]
    )
    filter_threshold: float | None = None  # None ⇒ percolation threshold
    filter_alpha: float = 0.05

    # Section 3: LRG multiscale
    tau_values: Sequence[float] | None = None
    normalized_laplacian: bool = True
    run_lrg: bool = True

    # Post-filter standard metrics
    run_standard_metrics: bool = True
    run_community_detection: bool = False
    run_rich_club: bool = False

    seed: int | None = None
