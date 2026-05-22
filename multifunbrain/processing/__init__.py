"""Network and signal processing (signed → unsigned, backbone, partial corr, temporal).

Each function lives in exactly one place; this package aggregates the public
API for ergonomic imports::

    from multifunbrain.processing import filter_absolute_threshold

Submodules:

- :mod:`multifunbrain.processing.filtering`: signed-to-unsigned network filters.
- :mod:`multifunbrain.processing.backbone`: statistically validated filters
  (disparity, LANS, MP-validated).
- :mod:`multifunbrain.processing.partial_correlation`: precision-matrix /
  partial-correlation estimation.
- :mod:`multifunbrain.processing.percolation`: weight-threshold for first-node
  detachment.
- :mod:`multifunbrain.processing.temporal`: Butterworth temporal filtering.
"""

from __future__ import annotations

from .backbone import filter_validated
from .filtering import (
    apply_all_filters,
    filter_absolute_threshold,
    filter_partial_correlation,
    filter_split_sign,
)
from .null_models import cached_surrogate_linkages, strength_preserving_rewire
from .partial_correlation import compute_precision_matrix
from .percolation import percolation_curve, percolation_threshold
from .temporal import band_filter

__all__ = [
    "apply_all_filters",
    "band_filter",
    "cached_surrogate_linkages",
    "compute_precision_matrix",
    "filter_absolute_threshold",
    "filter_partial_correlation",
    "filter_split_sign",
    "filter_validated",
    "percolation_curve",
    "percolation_threshold",
    "strength_preserving_rewire",
]
