"""Metadata-aware loader for the April 2026 correlation-matrix batch.

The April dataset is stratified by

- **contrast** (``co2`` | ``rest``)
- **processing variant** (``bpfBOLD``, ``bpfVASO``, ``MIRNoise_bold``,
  ``optcom_bold``, ``optcomMIRDenoised_bold``)
- **band** (``s4`` | ``s5`` | ``sstar``)
- **subject** (``sub-XXXXXXXX``, 6 subjects)

across three aggregation levels:

- ``global`` — one matrix per (contrast, processing), aggregated.
- ``band`` — per (contrast, processing, band) aggregate.
- ``patient`` — per (subject, contrast, processing, band) matrix.

Returns metadata-tagged :class:`AprilEntry` records so the
stratification metadata survives the pipeline (it would otherwise be
lost when flattened by :func:`multifunbrain.pipeline.discover_matrices`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..io.corrmatrix import load_correlation_matrix

__all__ = [
    "AprilEntry",
    "DEFAULT_APRIL_ROOT",
    "PROCESSING_VARIANTS",
    "CONTRASTS",
    "BANDS",
    "LEVELS",
    "N_PARCELS",
    "N_TIMEPOINTS_PER_PROCESSING",
    "discover_april",
    "entries_to_dataframe",
    "gamma_for",
    "label_for",
    "load_entry",
]

DEFAULT_APRIL_ROOT = Path(
    "data/correlation_mat_april_data"
)

CONTRASTS = ("co2", "rest")
PROCESSING_VARIANTS = (
    "bpfBOLD",
    "bpfVASO",
    "MIRNoise_bold",
    "optcom_bold",
    "optcomMIRDenoised_bold",
)
BANDS = ("s4", "s5", "sstar")
LEVELS: tuple[Literal["global", "band", "patient"], ...] = ("global", "band", "patient")

# Atlas: Schaefer 2018, 100 parcels × 17 networks → p = 100.
# n_timepoints measured from the raw .ts.1D files.
N_PARCELS: int = 100
N_TIMEPOINTS_PER_PROCESSING: dict[str, int] = {
    "bpfBOLD": 442,
    "bpfVASO": 442,
    "optcom_bold": 597,
    "optcomMIRDenoised_bold": 597,
    "MIRNoise_bold": 597,
}

_LEVEL_DIR = {
    "global": "freq-contrast-global",
    "band": "freq-contrast-inter",
    "patient": "freq-user-constrast-inter",  # collaborator-supplied typo kept intentionally
}

_EXCLUDE_NAMES = {".DS_Store", "All_Emd_diz.pkl"}
_EXCLUDE_DIRS = {"hist_C_I"}

_PROC_ALT = "|".join(re.escape(p) for p in PROCESSING_VARIANTS)
_CONTRAST_ALT = "|".join(CONTRASTS)
_BAND_ALT = "|".join(BANDS)
_SUBJECT_PATTERN = re.compile(r"^sub-[\w]+$")

_GLOBAL_FILE_RE = re.compile(
    rf"^(?P<contrast>{_CONTRAST_ALT})_(?P<processing>{_PROC_ALT})_GLOBAL\.pkl$"
)
_BAND_FILE_RE = re.compile(
    rf"^(?P<contrast>{_CONTRAST_ALT})_(?P<processing>{_PROC_ALT})_(?P<band>{_BAND_ALT})\.pkl$"
)
_PATIENT_FILE_RE = re.compile(
    rf"^(?P<subject>sub-[\w]+)_(?P<contrast>{_CONTRAST_ALT})_(?P<processing>{_PROC_ALT})_(?P<band>{_BAND_ALT})\.pkl$"
)


@dataclass(frozen=True)
class AprilEntry:
    """One correlation-matrix record from the April batch."""

    level: Literal["global", "band", "patient"]
    contrast: Literal["co2", "rest"]
    processing: str
    band: str | None
    subject: str | None
    path: Path


def label_for(entry: AprilEntry) -> str:
    """Pipeline label string for an entry.

    Examples
    --------
    ``"global/co2_bpfBOLD"``
    ``"band/co2_bpfBOLD/s4"``
    ``"patient/sub-00246757/co2_bpfBOLD/s4"``
    """
    proc_tag = f"{entry.contrast}_{entry.processing}"
    if entry.level == "global":
        return f"global/{proc_tag}"
    if entry.level == "band":
        return f"band/{proc_tag}/{entry.band}"
    return f"patient/{entry.subject}/{proc_tag}/{entry.band}"


def load_entry(entry: AprilEntry) -> np.ndarray:
    """Load the correlation matrix referenced by an entry.

    Thin wrapper around :func:`multifunbrain.io.corrmatrix.load_correlation_matrix`.
    Add a structural adapter here if the collaborator pickles turn out
    to be dict-wrapped (rather than bare arrays) — single place to do it.
    """
    return load_correlation_matrix(entry.path)


def gamma_for(entry: AprilEntry) -> float | None:
    """Marchenko-Pastur aspect ratio ``gamma = p / n`` for an entry.

    Looks up ``n_timepoints`` for the entry's processing variant from
    :data:`N_TIMEPOINTS_PER_PROCESSING` and divides
    :data:`N_PARCELS` by it. Returns ``None`` if the processing variant
    is not recognised (the pipeline then runs without MP overlay).
    """
    n_t = N_TIMEPOINTS_PER_PROCESSING.get(entry.processing)
    if n_t is None or n_t <= 0:
        return None
    return N_PARCELS / n_t


def _parse_global(path: Path) -> AprilEntry | None:
    m = _GLOBAL_FILE_RE.match(path.name)
    if not m:
        return None
    proc_tag = f"{m['contrast']}_{m['processing']}"
    if path.parent.name != proc_tag:
        return None
    return AprilEntry(
        level="global",
        contrast=m["contrast"],
        processing=m["processing"],
        band=None,
        subject=None,
        path=path,
    )


def _parse_band(path: Path) -> AprilEntry | None:
    m = _BAND_FILE_RE.match(path.name)
    if not m:
        return None
    proc_tag = f"{m['contrast']}_{m['processing']}"
    if path.parent.name != proc_tag:
        return None
    return AprilEntry(
        level="band",
        contrast=m["contrast"],
        processing=m["processing"],
        band=m["band"],
        subject=None,
        path=path,
    )


def _parse_patient(path: Path) -> AprilEntry | None:
    m = _PATIENT_FILE_RE.match(path.name)
    if not m:
        return None
    proc_tag = f"{m['contrast']}_{m['processing']}"
    if path.parent.name != proc_tag:
        return None
    subject_dir = path.parent.parent.name
    if not _SUBJECT_PATTERN.match(subject_dir) or subject_dir != m["subject"]:
        return None
    return AprilEntry(
        level="patient",
        contrast=m["contrast"],
        processing=m["processing"],
        band=m["band"],
        subject=m["subject"],
        path=path,
    )


_PARSERS = {
    "global": _parse_global,
    "band": _parse_band,
    "patient": _parse_patient,
}


def discover_april(
    root: str | Path = DEFAULT_APRIL_ROOT,
    levels: Iterable[Literal["global", "band", "patient"]] | None = None,
) -> list[AprilEntry]:
    """Walk the April directory tree and return tagged :class:`AprilEntry` records.

    Parameters
    ----------
    root : str or Path
        Root of the April batch. Defaults to
        ``data/correlation_mat_april_data`` (relative to CWD).
    levels : iterable or None
        Which aggregation levels to include. ``None`` (default) includes all.

    Returns
    -------
    list of AprilEntry
        Sorted by ``(level, contrast, processing, band, subject)``.

    Notes
    -----
    Excludes ``.DS_Store``, ``All_Emd_diz.pkl``, and the ``hist_C_I/``
    visualisation subdirectory. Files whose name does not match the
    expected pattern for their level are skipped silently — they are
    not part of the documented batch.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    selected = tuple(levels) if levels is not None else LEVELS
    entries: list[AprilEntry] = []

    for level in selected:
        if level not in _LEVEL_DIR:
            raise ValueError(f"Unknown level: {level!r}. Valid: {LEVELS}")
        level_dir = root / _LEVEL_DIR[level]
        if not level_dir.is_dir():
            continue
        parser = _PARSERS[level]
        for pkl in level_dir.rglob("*.pkl"):
            if pkl.name in _EXCLUDE_NAMES:
                continue
            if any(part in _EXCLUDE_DIRS for part in pkl.parts):
                continue
            entry = parser(pkl)
            if entry is not None:
                entries.append(entry)

    entries.sort(
        key=lambda e: (
            e.level,
            e.contrast,
            e.processing,
            e.band or "",
            e.subject or "",
        )
    )
    return entries


def entries_to_dataframe(entries: list[AprilEntry]) -> pd.DataFrame:
    """Manifest DataFrame for a list of entries.

    Columns: ``level``, ``contrast``, ``processing``, ``band``,
    ``subject``, ``path``, ``label``.
    """
    rows = [
        {
            "level": e.level,
            "contrast": e.contrast,
            "processing": e.processing,
            "band": e.band,
            "subject": e.subject,
            "path": str(e.path),
            "label": label_for(e),
        }
        for e in entries
    ]
    return pd.DataFrame(rows, columns=["level", "contrast", "processing", "band", "subject", "path", "label"])
