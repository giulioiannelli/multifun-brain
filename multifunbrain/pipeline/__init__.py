"""Three-section pipeline orchestration.

The pipeline takes a raw signed correlation matrix and produces a
structured :class:`PipelineResult` with:

1. **Descriptive analysis** of the raw signed dense network.
2. **Network filtering** to one or more unsigned tractable networks.
3. **Standard metrics + LRG multiscale clustering** on each filtered network.

Submodules:

- :mod:`multifunbrain.pipeline.config`: :class:`PipelineConfig` dataclass.
- :mod:`multifunbrain.pipeline.result`: :class:`PipelineResult` + sanitiser.
- :mod:`multifunbrain.pipeline.runner`: ``run_pipeline``, ``run_pipeline_batch``,
  ``run_pipeline_directory``.
- :mod:`multifunbrain.pipeline.discovery`: ``discover_matrices``.

For loading saved results, see :mod:`multifunbrain.io.results`. The
:class:`ResultsCollection` and :func:`load_results` are re-exported here
for legacy ``from multifunbrain.pipeline import load_results``.
"""

from __future__ import annotations

from ..io.results import ResultsCollection, load_results
from .config import PipelineConfig
from .discovery import discover_matrices
from .multiscale import MultiscaleResult, run_multiscale_lrg
from .result import PipelineResult
from .runner import run_pipeline, run_pipeline_batch, run_pipeline_directory

__all__ = [
    "MultiscaleResult",
    "PipelineConfig",
    "PipelineResult",
    "ResultsCollection",
    "discover_matrices",
    "load_results",
    "run_multiscale_lrg",
    "run_pipeline",
    "run_pipeline_batch",
    "run_pipeline_directory",
]
