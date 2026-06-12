"""Serializers for Section-1 (descriptive) plots: correlation heatmap, ...

Each function takes a :class:`~multifunbrain.pipeline.result.PipelineResult` and
returns a JSON-safe dict the frontend renders with Plotly. Node identity always
flows through :mod:`dashboard.backend.remap` so atlas hover labels stay aligned
after dead-region dropping.
"""

from __future__ import annotations

import numpy as np

from ..encode import clean
from ..remap import surviving_labels


def heatmap_spec(result) -> dict:
    """Prepared correlation matrix + atlas node descriptors for an interactive heatmap."""
    if result.corr_prepared is None:
        return {"kind": "heatmap", "label": result.label, "error": "no corr_prepared stored"}

    matrix = np.asarray(result.corr_prepared)
    n = int(matrix.shape[0])
    nodes = surviving_labels(result)
    if len(nodes) != n:
        # Defensive: atlas/matrix size mismatch -> fall back to generic labels.
        nodes = [
            {
                "index": i,
                "name": f"node-{i}",
                "short": f"node-{i}",
                "hemisphere": "?",
                "network": "?",
                "color": "#888888",
            }
            for i in range(n)
        ]

    return clean(
        {
            "kind": "heatmap",
            "label": result.label,
            "n": n,
            "z": matrix,
            "nodes": nodes,
            "names": [nd["short"] for nd in nodes],
            "networks": [nd["network"] for nd in nodes],
            "colors": [nd["color"] for nd in nodes],
            "zmin": -1.0,
            "zmax": 1.0,
        }
    )
