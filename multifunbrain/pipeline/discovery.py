"""Discover correlation-matrix files under a directory tree."""

from __future__ import annotations

from pathlib import Path

__all__ = ["discover_matrices", "label_from_path", "SUPPORTED_EXTENSIONS"]

SUPPORTED_EXTENSIONS = frozenset({".npy", ".npz", ".csv", ".txt", ".pkl"})


def discover_matrices(
    directory: str | Path,
    extensions: set[str] | None = None,
    recursive: bool = True,
    pattern: str | None = None,
) -> list[Path]:
    """Find all correlation matrix files under *directory*.

    Parameters
    ----------
    directory : str or Path
        Root directory to search.
    extensions : set of str or None
        File extensions to include (default: ``.npy .npz .csv .txt .pkl``).
    recursive : bool
        If ``True`` (default), search subdirectories recursively.
    pattern : str or None
        If given, only include files whose name contains *pattern*
        (case-insensitive).

    Returns
    -------
    list of Path
        Sorted list of discovered file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    exts = extensions or SUPPORTED_EXTENSIONS
    glob_method = directory.rglob if recursive else directory.glob
    files: list[Path] = []
    for ext in sorted(exts):
        files.extend(glob_method(f"*{ext}"))
    if pattern:
        pat = pattern.lower()
        files = [f for f in files if pat in f.name.lower()]
    return sorted(files)


def label_from_path(path: Path, root: Path) -> str:
    """Derive a human-readable label from a file path relative to *root*."""
    rel = path.relative_to(root)
    parts = list(rel.parent.parts) + [rel.stem]
    return "/".join(parts)
