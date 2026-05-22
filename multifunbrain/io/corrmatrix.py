"""File I/O for correlation matrices.

Single-responsibility module: read correlation matrices from disk
(``.npy``, ``.npz``, ``.csv``, ``.txt``, ``.pkl``) into a numpy array.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

__all__ = ["load_correlation_matrix"]


def load_correlation_matrix(path: str | Path) -> np.ndarray:
    """Load a correlation matrix from ``.npy``, ``.npz``, ``.csv``, ``.txt``, or ``.pkl`` files.

    Parameters
    ----------
    path : str or Path
        File path to load.

    Returns
    -------
    np.ndarray
        The loaded correlation matrix.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data[list(data.keys())[0]]
    if path.suffix in {".csv", ".txt"}:
        delimiter = "," if path.suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)
    if path.suffix == ".pkl":
        with open(path, "rb") as fh:
            return np.asarray(pickle.load(fh))

    raise ValueError(f"Unsupported file type: {path.suffix}")
