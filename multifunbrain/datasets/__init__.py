"""Dataset-specific loaders.

Each subdataset has its own metadata-aware module:

- :mod:`multifunbrain.datasets.april`: April 2026 correlation-matrix
  batch (CO2 vs rest, 5 processing variants, 3 bands, 6 subjects,
  3 aggregation levels).
"""

from __future__ import annotations

from . import april
from .april import (
    AprilEntry,
    discover_april,
    entries_to_dataframe,
    gamma_for,
    label_for,
    load_entry,
)

__all__ = [
    "AprilEntry",
    "april",
    "discover_april",
    "entries_to_dataframe",
    "gamma_for",
    "label_for",
    "load_entry",
]
