"""I/O for pipeline results: load/save :class:`PipelineResult` collections.

Pickled ``results.pkl`` files (a list of :class:`PipelineResult`) are
loaded into a :class:`ResultsCollection` with label-based indexing,
iteration, and summary-table aggregation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

# Avoid a runtime circular import: pipeline.result -> io.results would
# create a cycle. ResultsCollection only needs ``PipelineResult`` for
# typing, so use a forward reference and import inside ``load_results``.

__all__ = ["ResultsCollection", "load_results"]


class ResultsCollection:
    """List-like container for :class:`PipelineResult` with label-based access.

    Supports integer indexing, iteration, ``len()``, and label-based
    lookup via ``results["my_label"]`` or ``results.labels``.

    Examples
    --------
    >>> results = load_results("pipeline_results/")
    >>> results[0]                    # first result
    >>> results["my_label"]           # by label
    >>> results.labels                # all labels
    >>> results.summary_table()       # combined summary across all results
    >>> for r in results:             # iterate
    ...     print(r.label)
    """

    def __init__(self, results) -> None:
        self._results = list(results)
        self._label_index: dict[str, int] = {}
        for i, r in enumerate(self._results):
            if r.label is not None:
                self._label_index[r.label] = i

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self._label_index.get(key)
            if idx is None:
                raise KeyError(f"No result with label {key!r}")
            return self._results[idx]
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __repr__(self) -> str:
        return f"ResultsCollection({len(self)} results)"

    @property
    def labels(self) -> list[str | None]:
        """List of labels in order."""
        return [r.label for r in self._results]

    def filter(self, pattern: str) -> ResultsCollection:
        """Return a new collection keeping only results whose label contains *pattern*."""
        pat = pattern.lower()
        return ResultsCollection(
            [r for r in self._results if r.label and pat in r.label.lower()]
        )

    def summary_table(self) -> pd.DataFrame:
        """Concatenated summary table across all results."""
        frames = [r.summary_table() for r in self._results if r.network_analyses]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def load_results(path: str | Path) -> ResultsCollection:
    """Load pipeline results from a previous run.

    Parameters
    ----------
    path : str or Path
        Either the ``results.pkl`` file directly, or the output directory
        that contains it (e.g. ``"pipeline_results/"``).

    Returns
    -------
    ResultsCollection
        Results with label-based indexing, iteration, and summary table.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "results.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    if isinstance(data, list):
        return ResultsCollection(data)
    return ResultsCollection([data])
