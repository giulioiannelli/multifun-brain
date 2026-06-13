"""File I/O for raw ROI **timecourses** (AFNI ``.ts.1D`` files).

The raw fMRI hand-off ships one ``.ts.1D`` text file per
``(subject, task/contrast, processing variant)`` — AFNI ``3dROIstats`` output,
laid out as ``n_timepoints`` rows × ``n_regions`` columns (the 100 Schaefer
parcels). This module is the single canonical home for **reading** those files
and **discovering** them on disk, plus parsing the BIDS-like AFNI filename into
``(subject, session, contrast, run, processing)`` metadata.

This is the raw-timeseries sibling of :mod:`multifunbrain.io.corrmatrix`
(correlation-matrix I/O). It is deliberately **not** in
:mod:`multifunbrain.datasets.april`: that loader handles the *correlation-matrix*
pickle batch; raw timecourses are a different artefact with a different layout.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CONTRASTS",
    "PROCESSING_VARIANTS",
    "TimecourseFile",
    "discover_timecourses",
    "load_timecourses",
    "parse_timecourse_filename",
]

CONTRASTS = ("co2", "rest")
PROCESSING_VARIANTS = (
    "bpfBOLD",
    "bpfVASO",
    "MIRNoise_bold",
    "optcom_bold",
    "optcomMIRDenoised_bold",
)

# Longest-first alternation so ``optcomMIRDenoised_bold`` is matched before the
# shorter ``optcom_bold`` prefix it shares.
_PROC_ALT = "|".join(
    sorted((re.escape(p) for p in PROCESSING_VARIANTS), key=len, reverse=True)
)
_FILENAME_RE = re.compile(
    r"(?P<subject>sub-[A-Za-z0-9]+)"
    r"_ses-(?P<session>[A-Za-z0-9]+)"
    r"_task-(?P<contrast>co2|rest)"
    r"_run-(?P<run>\d+)"
    rf".*?_desc-(?P<processing>{_PROC_ALT})\.ts\.1D$"
)


@dataclass(frozen=True)
class TimecourseFile:
    """One discovered raw-timecourse file and its parsed metadata."""

    subject: str  # "sub-XXXXXXXX"
    session: str  # "ses-YYYYMMDD"
    contrast: str  # task label: "co2" | "rest"
    run: str  # zero-padded run index as written in the filename
    processing: str  # one of PROCESSING_VARIANTS
    path: Path


def parse_timecourse_filename(name: str) -> dict | None:
    """Parse an AFNI ``.ts.1D`` filename into metadata, or ``None`` if it doesn't match.

    Recognises the documented hand-off pattern
    ``..._sub-<id>_ses-<date>_task-<co2|rest>_run-<nn>..._desc-<variant>.ts.1D``.
    """
    m = _FILENAME_RE.search(name)
    if not m:
        return None
    return {
        "subject": m["subject"],
        "session": m["session"],
        "contrast": m["contrast"],
        "run": m["run"],
        "processing": m["processing"],
    }


def load_timecourses(
    path: str | Path, *, region_major: bool = True
) -> np.ndarray:
    """Load an AFNI ``.ts.1D`` ROI-timecourse file into a 2-D array.

    The on-disk layout is ``n_timepoints`` rows × ``n_regions`` columns.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ts.1D`` (whitespace-delimited text) file.
    region_major : bool, default True
        If True, return shape ``(n_regions, n_timepoints)`` (the convention used
        by :func:`multifunbrain.analysis.corrnet.compute_correlation_matrix`).
        If False, return the file's native ``(n_timepoints, n_regions)``.

    Returns
    -------
    np.ndarray
        Float array of timecourses.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not parse as a 2-D numeric array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D timecourse array in {path.name}, got shape {arr.shape}"
        )
    return arr.T if region_major else arr


def discover_timecourses(
    root: str | Path,
    *,
    contrasts: Iterable[str] | None = None,
    processings: Iterable[str] | None = None,
) -> list[TimecourseFile]:
    """Walk *root* for ``.ts.1D`` files and return parsed :class:`TimecourseFile` records.

    Files whose name does not match the documented pattern are skipped silently
    (they are not part of the batch). Results are sorted by
    ``(subject, contrast, processing, run)``.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    keep_c = set(contrasts) if contrasts is not None else None
    keep_p = set(processings) if processings is not None else None

    out: list[TimecourseFile] = []
    for path in sorted(root.rglob("*.ts.1D")):
        meta = parse_timecourse_filename(path.name)
        if meta is None:
            continue
        if keep_c is not None and meta["contrast"] not in keep_c:
            continue
        if keep_p is not None and meta["processing"] not in keep_p:
            continue
        out.append(TimecourseFile(path=path, **meta))

    out.sort(key=lambda e: (e.subject, e.contrast, e.processing, e.run))
    return out
