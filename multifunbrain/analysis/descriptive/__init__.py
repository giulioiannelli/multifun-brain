"""Descriptive analysis (Section 1): characterise a raw signed correlation matrix.

Submodules:

- :mod:`multifunbrain.analysis.descriptive.weights`: edge-weight distribution.
- :mod:`multifunbrain.analysis.descriptive.spectrum`: eigenvalue spectrum + RMT.
- :mod:`multifunbrain.analysis.descriptive.signed`: signed-Laplacian metrics.
- :mod:`multifunbrain.analysis.descriptive.report`: aggregate descriptive report.

For partial-correlation / precision-matrix estimation, see
:mod:`multifunbrain.processing.partial_correlation`. It is re-exported
here as ``compute_precision_matrix`` so legacy
``from multifunbrain.analysis.descriptive import compute_precision_matrix``
keeps working.
"""

from __future__ import annotations

from ...processing.partial_correlation import compute_precision_matrix
from .report import descriptive_report
from .signed import (
    signed_laplacian_analysis,
    signed_laplacian_and_spectrum,
    signed_network_metrics,
)
from .spectrum import correlation_spectrum_analysis
from .weights import weight_distribution_analysis

__all__ = [
    "compute_precision_matrix",
    "correlation_spectrum_analysis",
    "descriptive_report",
    "signed_laplacian_analysis",
    "signed_laplacian_and_spectrum",
    "signed_network_metrics",
    "weight_distribution_analysis",
]
