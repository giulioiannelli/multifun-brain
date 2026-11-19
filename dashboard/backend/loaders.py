"""Lazy, memoised loading of result bundles.

``load_results`` (the sanctioned pickle call site) is wrapped in an LRU keyed by
``(directory, results.pkl mtime)`` so repeat views are instant but a re-run that
rewrites ``results.pkl`` is picked up automatically. ``invalidate`` clears the
cache after ingestion writes new results.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from multifunbrain.io import load_results


@lru_cache(maxsize=64)
def _load_cached(dir_str: str, _mtime: float):
    return load_results(dir_str)


def get_results(dataset_dir: str | Path):
    """Return the (cached) :class:`ResultsCollection` for a dataset directory."""
    dataset_dir = Path(dataset_dir)
    pkl = dataset_dir / "results.pkl"
    return _load_cached(str(dataset_dir), pkl.stat().st_mtime)


def invalidate() -> None:
    """Drop all cached collections (call after a new bundle is written)."""
    _load_cached.cache_clear()
