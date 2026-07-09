"""Schaefer-2018 atlas parsing + parcel centroids.

Reads the ``*_order.txt`` file (``index  name  R  G  B  0`` per parcel) into
:class:`Region` records used to label heatmap/network/3-D nodes. Parcel centroid
MNI coordinates (for the 3-D brain) are derived once from the parcellation NIfTI
via nilearn and cached to disk.

Atlas note: the data-layout guide mentions "17 networks" but only the
7-network order file ships in the repo; we parse whatever file is configured and
expose its actual network labels.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class Region:
    """One atlas parcel."""

    index: int  # 0-based parcel index (matches matrix row order before dropping)
    name: str  # raw, e.g. "7Networks_LH_Vis_1"
    short: str  # without the leading "NNetworks_", e.g. "LH_Vis_1"
    hemisphere: str  # "LH" | "RH"
    network: str  # "Vis", "SomMot", ...
    color: str  # "#rrggbb"


@lru_cache(maxsize=4)
def load_atlas(order_file: str | None = None) -> tuple[Region, ...]:
    """Parse the Schaefer order file into a tuple of :class:`Region` (0-indexed)."""
    path = Path(order_file) if order_file else config.ATLAS_ORDER_FILE
    regions: list[Region] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        idx = int(parts[0]) - 1  # file is 1-based; matrices are 0-based
        name = parts[1]
        r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
        toks = name.split("_")
        hemisphere = toks[1] if len(toks) > 1 else "?"
        network = toks[2] if len(toks) > 2 else "?"
        short = name.split("_", 1)[1] if "_" in name else name
        regions.append(
            Region(idx, name, short, hemisphere, network, f"#{r:02x}{g:02x}{b:02x}")
        )
    return tuple(regions)


def region_count() -> int:
    return len(load_atlas())


@lru_cache(maxsize=2)
def harvard_oxford_names(xml_file: str | None = None) -> tuple[str, ...]:
    """Region names (in 0-based index order) from the HarvardOxford-Cortical
    FSL atlas XML — the channel labels for the 48-parcel raw set.

    The XML is ``<label index="i" …>Name</label>`` per parcel; we return the
    names sorted by ``index``. Returns an empty tuple when the file is absent or
    unparseable (``data/`` is gitignored, so the XML ships with the collaborator
    data, not the repo), letting callers fall back to generic ``region-N``
    labels instead of failing.
    """
    path = Path(xml_file) if xml_file else config.HARVARD_OXFORD_XML
    if not path.exists():
        return ()
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return ()
    labels: list[tuple[int, str]] = []
    for label in root.findall(".//label"):
        idx = label.get("index")
        if idx is None:
            continue
        labels.append((int(idx), (label.text or "").strip()))
    labels.sort(key=lambda t: t[0])
    return tuple(name for _, name in labels)


@lru_cache(maxsize=1)
def parcel_centroids() -> np.ndarray:
    """(n_parcels, 3) MNI coordinates of parcel centroids, cached to disk.

    Imports nilearn lazily so the dashboard's non-3-D paths never pay for it.
    """
    cache = config.CACHE_DIR / "schaefer_centroids.npy"
    if cache.exists():
        return np.load(cache)
    from nilearn.plotting import find_parcellation_cut_coords

    coords = np.asarray(find_parcellation_cut_coords(str(config.ATLAS_NIFTI)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, coords)
    return coords
