"""File I/O for the multifunbrain package.

Submodules:

- :mod:`multifunbrain.io.corrmatrix`: load correlation matrices from disk.
- :mod:`multifunbrain.io.timeseries`: read raw ROI timecourses (``.ts.1D``).
- :mod:`multifunbrain.io.results`: load saved :class:`PipelineResult`
  collections (``results.pkl``).
"""

from __future__ import annotations

from .corrmatrix import load_correlation_matrix
from .results import ResultsCollection, load_results
from .timeseries import (
    TimecourseFile,
    discover_timecourses,
    load_timecourses,
    parse_timecourse_filename,
)

__all__ = [
    "ResultsCollection",
    "TimecourseFile",
    "discover_timecourses",
    "load_correlation_matrix",
    "load_results",
    "load_timecourses",
    "parse_timecourse_filename",
]
