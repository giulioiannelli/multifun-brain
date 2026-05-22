"""Correlation-matrix preprocessing (pre-analysis cleaning).

Submodules:
- :mod:`multifunbrain.preprocessing.dead_regions`: detect zero-variance regions.
- :mod:`multifunbrain.preprocessing.prepare`: symmetrise / clean / clip.
- :mod:`multifunbrain.preprocessing.denoising`: Marchenko-Pastur denoising.
"""

from __future__ import annotations

from .dead_regions import detect_dead_regions
from .denoising import marchenko_pastur_denoise, marchenko_pastur_density
from .prepare import prepare_correlation_matrix

__all__ = [
    "detect_dead_regions",
    "marchenko_pastur_denoise",
    "marchenko_pastur_density",
    "prepare_correlation_matrix",
]
