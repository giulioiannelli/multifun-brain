"""Dead-region re-indexing — the single source of truth for node identity.

The pipeline drops dead brain regions during preparation, so ``corr_prepared``,
filtered graphs, and LRG partitions all index *survivors* in ascending original
order. Atlas labels and centroids, however, are length ``n_regions_original``.
Every node-labelled spec (heatmap, network, dendrogram, brain3d) must map
survivor positions back to original atlas indices here — never inline — so hover
names and 3-D nodes stay aligned.
"""

from __future__ import annotations

import numpy as np

from . import atlas as atlas_mod


def survivor_indices(n_original: int, dropped) -> list[int]:
    """Original 0-based indices that survived dead-region dropping, in order."""
    dropped_set = {int(i) for i in np.asarray(dropped).ravel().tolist()}
    return [i for i in range(int(n_original)) if i not in dropped_set]


def _generic_node(orig: int) -> dict:
    return {
        "index": orig,
        "name": f"node-{orig}",
        "short": f"node-{orig}",
        "hemisphere": "?",
        "network": "?",
        "color": "#888888",
    }


def surviving_labels(result) -> list[dict]:
    """Atlas descriptors aligned to ``result.corr_prepared`` / filtered-graph rows."""
    survivors = survivor_indices(result.n_regions_original, result.dropped_regions)
    regions = atlas_mod.load_atlas()
    out: list[dict] = []
    for orig in survivors:
        if 0 <= orig < len(regions):
            reg = regions[orig]
            out.append(
                {
                    "index": orig,
                    "name": reg.name,
                    "short": reg.short,
                    "hemisphere": reg.hemisphere,
                    "network": reg.network,
                    "color": reg.color,
                }
            )
        else:
            out.append(_generic_node(orig))
    return out


def surviving_coords(result) -> np.ndarray:
    """(n_survivors, 3) MNI centroids aligned to survivor order."""
    survivors = survivor_indices(result.n_regions_original, result.dropped_regions)
    coords = atlas_mod.parcel_centroids()
    return coords[survivors]
