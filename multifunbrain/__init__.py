"""Utility package for hierarchical modular brain network analysis."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import analysis, generation, notebook, visualization
from .core import band_filter, hello_brain, marchenko_pastur_density

try:  # pragma: no cover - fallback when package metadata is unavailable
    __version__ = version("multifunbrain")
except PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.2.0"

__all__ = [
    "analysis",
    "band_filter",
    "generation",
    "hello_brain",
    "notebook",
    "marchenko_pastur_density",
    "visualization",
    "__version__",
]
