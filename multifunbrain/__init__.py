"""Utility package for hierarchical modular brain network analysis.

Scope-based submodules (each function/class has one canonical home):

- :mod:`multifunbrain.io`: file I/O (correlation matrices, pipeline results).
- :mod:`multifunbrain.preprocessing`: matrix cleaning, dead-region detection,
  MP denoising.
- :mod:`multifunbrain.processing`: signed-to-unsigned filtering, backbone
  extraction, partial correlation, temporal filtering.
- :mod:`multifunbrain.analysis`: descriptive metrics, LRG multiscale,
  standard network metrics, partition comparison.
- :mod:`multifunbrain.pipeline`: orchestrators (config, runner, discovery, result).
- :mod:`multifunbrain.datasets`: dataset-specific loaders (April batch, etc.).
- :mod:`multifunbrain.visualization`: plotting helpers.
- :mod:`multifunbrain.generation`: synthetic-network generators.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import (
    analysis,
    datasets,
    generation,
    io,
    notebook,
    pipeline,
    preprocessing,
    processing,
    visualization,
)
from .core import band_filter, hello_brain, marchenko_pastur_density
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    ResultsCollection,
    discover_matrices,
    load_results,
    run_pipeline,
    run_pipeline_batch,
    run_pipeline_directory,
)

try:  # pragma: no cover - fallback when package metadata is unavailable
    __version__ = version("multifunbrain")
except PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.2.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "ResultsCollection",
    "__version__",
    "analysis",
    "band_filter",
    "datasets",
    "discover_matrices",
    "generation",
    "hello_brain",
    "io",
    "load_results",
    "marchenko_pastur_density",
    "notebook",
    "pipeline",
    "preprocessing",
    "processing",
    "run_pipeline",
    "run_pipeline_batch",
    "run_pipeline_directory",
    "visualization",
]
