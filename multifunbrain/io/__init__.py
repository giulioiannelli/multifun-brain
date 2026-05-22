"""File I/O for the multifunbrain package.

Submodules:

- :mod:`multifunbrain.io.corrmatrix`: load correlation matrices from disk.
- :mod:`multifunbrain.io.results`: load saved :class:`PipelineResult`
  collections (``results.pkl``).
"""

from __future__ import annotations

from .corrmatrix import load_correlation_matrix
from .results import ResultsCollection, load_results

__all__ = ["ResultsCollection", "load_correlation_matrix", "load_results"]
