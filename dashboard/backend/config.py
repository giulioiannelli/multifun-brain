"""Paths and runtime configuration for the dashboard backend.

All roots are env-overridable so the same app can run locally (defaults,
relative to the repo) or behind a server (explicit env vars) with no code
change. ``RESULTS_ROOT`` is the directory scanned for result bundles;
``DATA_ROOT`` is the allowlist boundary for ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path

# dashboard/backend/config.py -> repo root is two parents up from dashboard/.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


# Directory tree of pipeline output bundles. Each dataset is a sub-directory
# that contains a ``results.pkl`` (discovered recursively).
RESULTS_ROOT = _path_env(
    "MFB_RESULTS_ROOT", REPO_ROOT / "data" / "correlation_matrices_results"
)

# Allowlist boundary for ingestion: only folders under here may be elaborated.
DATA_ROOT = _path_env("MFB_DATA_ROOT", REPO_ROOT / "data")

# Raw ROI timecourses (AFNI ``.ts.1D``) for the Signal tab. One directory per
# subject, discovered recursively.
RAW_DATA_ROOT = _path_env("MFB_RAW_DATA_ROOT", REPO_ROOT / "data" / "raw_data")

# Schaefer atlas assets (region names + parcellation volume for 3-D).
ATLAS_DIR = _path_env("MFB_ATLAS_DIR", REPO_ROOT / "data" / "schaefer_2018")
ATLAS_ORDER_FILE = ATLAS_DIR / "Schaefer2018_100Parcels_7Networks_order.txt"
ATLAS_NIFTI = (
    ATLAS_DIR / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
)

# On-disk cache for computed plot specs / centroids.
CACHE_DIR = _path_env(
    "MFB_DASHBOARD_CACHE", REPO_ROOT / "dashboard" / ".cache"
)

# Built React bundle (served as static files when present).
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
