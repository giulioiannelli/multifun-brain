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
    "SAMPLING_INTERVAL_SECONDS",
    "TimecourseFile",
    "discover_timecourses",
    "load_timecourses",
    "parse_timecourse_filename",
    "sampling_rate",
]

CONTRASTS = ("co2", "rest")
PROCESSING_VARIANTS = (
    "bpfBOLD",
    "bpfVASO",
    "MIRNoise_bold",
    "optcom_bold",
    "optcomMIRDenoised_bold",
)

# Acquisition repetition time (TR, seconds) per processing variant, from the
# April hand-off (Daniele's ``desc_fs_map``). The band-pass (``bpf*``) variants
# are sampled slower than the optimally-combined / MIR variants; both groups
# cover ~10 min scans (bpf 442×1.353 s ≈ optcom 597×0.98 s). Characteristic
# frequencies are reported in Hz via ``fs = 1/TR`` — the units Daniele's fixed
# slow-oscillation bands (s5/s4/s*) are defined in. TR is variant-specific, so a
# single cycles/sample → Hz scale does **not** reconcile the two.
SAMPLING_INTERVAL_SECONDS = {
    "bpfBOLD": 1.353,
    "bpfVASO": 1.353,
    "MIRNoise_bold": 0.98,
    "optcom_bold": 0.98,
    "optcomMIRDenoised_bold": 0.98,
}


def sampling_rate(processing: str) -> float:
    """Sampling frequency ``fs = 1/TR`` (Hz) for a processing variant.

    Falls back to ``1.0`` (i.e. cycles/sample) for an unknown variant so callers
    degrade gracefully instead of raising.
    """
    tr = SAMPLING_INTERVAL_SECONDS.get(processing)
    return 1.0 / tr if tr else 1.0


# Longest-first alternation so ``optcomMIRDenoised_bold`` is matched before the
# shorter ``optcom_bold`` prefix it shares.
_PROC_ALT = "|".join(
    sorted((re.escape(p) for p in PROCESSING_VARIANTS), key=len, reverse=True)
)
# BIDS scheme (April batch): subject + task/contrast + run + desc are in the name.
_FILENAME_RE = re.compile(
    r"(?P<subject>sub-[A-Za-z0-9]+)"
    r"_ses-(?P<session>[A-Za-z0-9]+)"
    r"_task-(?P<contrast>co2|rest)"
    r"_run-(?P<run>\d+)"
    rf".*?_desc-(?P<processing>{_PROC_ALT})\.ts\.1D$"
)

# Atlas prefixes that begin every filename; map to a short id.
_ATLAS_PREFIX_ID = {
    "Schaefer2018_100Parcels_17Networks": "schaefer100",
    "HarvardOxford_48Parcels": "harvardoxford48",
}
# Older "kw" scheme (Nov-2025 / HarvardOxford batches): no subject/task/run in
# the name — just ``<atlas-prefix>_<modality-token>.ts.1D`` (subject comes from
# the parent ``sub-…`` folder). The modality token (e.g. ``kwBOLD4D``,
# ``clean_kwfurN_Bold``, ``kwoptcomMIRDenoised_bold``) is the ``processing`` facet.
_KW_RE = re.compile(
    r"^(?P<atlas>"
    + "|".join(re.escape(a) for a in _ATLAS_PREFIX_ID)
    + r")_(?P<processing>.+)\.ts\.1D$"
)


def _atlas_id(name: str) -> str | None:
    for prefix, aid in _ATLAS_PREFIX_ID.items():
        if name.startswith(prefix):
            return aid
    return None


@dataclass(frozen=True)
class TimecourseFile:
    """One discovered raw-timecourse file and its parsed metadata.

    ``session`` / ``contrast`` / ``run`` are ``None`` for the older ``kw…``
    scheme (single condition, no BIDS entities) — there ``subject`` comes from
    the parent ``sub-…`` directory. ``processing`` is the BIDS desc variant or
    the ``kw…`` modality token; ``atlas`` is the short atlas id.
    """

    subject: str  # "sub-XXXXXXXX"
    processing: str  # BIDS desc variant, or kw modality token
    path: Path
    session: str | None = None
    contrast: str | None = None  # "co2" | "rest", or None (kw scheme)
    run: str | None = None
    atlas: str | None = None  # "schaefer100" | "harvardoxford48"


def parse_timecourse_filename(name: str) -> dict | None:
    """Parse an AFNI ``.ts.1D`` filename into metadata, or ``None`` if no scheme matches.

    Recognises two schemes: the BIDS hand-off
    ``..._sub-<id>_ses-<date>_task-<co2|rest>_run-<nn>..._desc-<variant>.ts.1D``
    (tried first), and the older ``<atlas-prefix>_<modality>.ts.1D`` (``kw…``)
    scheme, whose ``subject`` is **not** in the name (``None`` here — discovery
    fills it from the parent folder) and which has no contrast/run.
    """
    m = _FILENAME_RE.search(name)
    if m:
        return {
            "subject": m["subject"],
            "session": m["session"],
            "contrast": m["contrast"],
            "run": m["run"],
            "processing": m["processing"],
            "atlas": _atlas_id(name),
        }
    k = _KW_RE.match(name)
    if k:
        return {
            "subject": None,
            "session": None,
            "contrast": None,
            "run": None,
            "processing": k["processing"],
            "atlas": _ATLAS_PREFIX_ID.get(k["atlas"]),
        }
    return None


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
        # Skip quarantined files (e.g. a ``discarded/`` subtree).
        if any(part == "discarded" for part in path.parts):
            continue
        # Skip empty files (some batches ship 0-byte ``*4D`` placeholders).
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        meta = parse_timecourse_filename(path.name)
        if meta is None:
            continue
        # kw-scheme files carry no subject in the name — take it from the parent
        # directory, which must be a ``sub-…`` folder.
        subject = meta["subject"]
        if subject is None:
            parent = path.parent.name
            if not parent.startswith("sub-"):
                continue
            subject = parent
        # Only filter by contrast when the file actually has one (kw scheme: None).
        if keep_c is not None and meta["contrast"] is not None and meta["contrast"] not in keep_c:
            continue
        if keep_p is not None and meta["processing"] not in keep_p:
            continue
        out.append(
            TimecourseFile(
                subject=subject,
                processing=meta["processing"],
                path=path,
                session=meta["session"],
                contrast=meta["contrast"],
                run=meta["run"],
                atlas=meta.get("atlas"),
            )
        )

    out.sort(key=lambda e: (e.subject, e.contrast or "", e.processing, e.run or ""))
    return out
