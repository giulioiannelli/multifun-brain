"""JSON-safety: convert pipeline objects to plain, finite Python.

Starlette's ``JSONResponse`` serialises with ``allow_nan=False``, so any
``NaN``/``Inf`` (e.g. ``avg_shortest_path`` on a disconnected graph) would raise.
``clean`` maps non-finite floats to ``None`` and numpy scalars/arrays to native
Python. Apply it at the serializer boundary so routes always return safe dicts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def clean(obj: Any) -> Any:
    """Recursively convert *obj* to JSON-safe Python (numpy -> native, NaN/Inf -> None)."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return obj
