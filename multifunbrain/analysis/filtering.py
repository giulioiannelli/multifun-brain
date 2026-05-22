"""Back-compat shim for :mod:`multifunbrain.processing`.

Filtering functions used to live here; they now have a single canonical
home in :mod:`multifunbrain.processing`. This module re-exports them so
existing imports (``from multifunbrain.analysis.filtering import ...``)
keep working.

New code should import from :mod:`multifunbrain.processing` directly.
"""

from __future__ import annotations

from ..processing.backbone import filter_validated
from ..processing.filtering import (
    apply_all_filters,
    filter_absolute_threshold,
    filter_partial_correlation,
    filter_split_sign,
)
from ..processing.percolation import percolation_threshold

__all__ = [
    "apply_all_filters",
    "filter_absolute_threshold",
    "filter_partial_correlation",
    "filter_split_sign",
    "filter_validated",
    "percolation_threshold",
]
