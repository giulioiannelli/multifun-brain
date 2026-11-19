"""Interactive glass-brain reprojections (nilearn HTML for an ``<iframe>``).

Two reprojections, both self-contained nilearn pages served as ``text/html``:

- :func:`partition_html` — the LRG partition at a chosen τ / dendrogram cut, each
  parcel coloured by its community. Colours mirror the dashboard dendrogram +
  clustered-network exactly (same union-find over the linkage, clusters ordered
  by leftmost-leaf position → the shared LRG palette), so the three views agree.
- :func:`scalar_filled_html` — a continuous per-ROI scalar (e.g. the calibrated
  cophenetic shift of the Compare tab) painted onto the Schaefer parcellation
  with a diverging colourmap, mirroring the collaborator's ``fig6`` brain rows.

Both degrade to a small HTML message when the atlas geometry is unavailable
(e.g. the 48-parcel HarvardOxford sets have no Schaefer centroids/volume).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.cluster.hierarchy import dendrogram

from . import config, remap

# Shared with the frontend LRG_PALETTE / backend _LRG_PALETTE so a community keeps
# one colour across dendrogram, clustered-network and brain.
LRG_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#393b79", "#637939", "#8c6d31",
    "#843c39", "#7b4173", "#5254a3", "#e7969c", "#a55194", "#6b6ecf",
]


def _error_page(message: str) -> str:
    return (
        "<!doctype html><html><body style='font-family:system-ui;color:#c62828;"
        "padding:16px'>" + message + "</body></html>"
    )


def leaf_cluster_ids(linkage: np.ndarray, cut_height: float) -> tuple[np.ndarray, int]:
    """Per-leaf community id at ``cut_height``, ordered as the dashboard colours.

    Mirrors the frontend ``colorDendrogram``: union-find over every merge below
    ``cut_height``, then clusters indexed by their leftmost leaf's position in the
    SciPy dendrogram leaf order. Returns ``(ids[n_leaves], n_clusters)`` with ids
    in ``0..n_clusters-1`` (the palette index).
    """
    Z = np.asarray(linkage, dtype=float)
    n = int(Z.shape[0] + 1)
    parent = list(range(2 * n - 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(Z.shape[0]):
        if Z[i, 2] <= cut_height:
            union(int(Z[i, 0]), int(Z[i, 1]))
            union(n + i, int(Z[i, 0]))

    leaves = dendrogram(Z, no_plot=True)["leaves"]
    leaf_pos = {int(lid): p for p, lid in enumerate(leaves)}
    root_min_pos: dict[int, int] = {}
    for leaf in range(n):
        r = find(leaf)
        p = leaf_pos.get(leaf, leaf)
        if r not in root_min_pos or p < root_min_pos[r]:
            root_min_pos[r] = p
    order = sorted(root_min_pos, key=lambda r: root_min_pos[r])
    color_of_root = {r: idx for idx, r in enumerate(order)}
    ids = np.array([color_of_root[find(leaf)] for leaf in range(n)], dtype=int)
    return ids, len(order)


@lru_cache(maxsize=1)
def _atlas_volume():
    """``(nib image, int data)`` of the Schaefer parcellation, or ``None``."""
    import nibabel as nib

    path = config.ATLAS_NIFTI
    if not path.exists():
        return None
    img = nib.load(str(path))
    return img, img.get_fdata().astype(int)


def _roi_image(values_by_atlas_idx: dict[int, float], background: float = np.nan):
    """Fill a copy of the parcellation with a per-parcel value (0-based idx)."""
    import nibabel as nib

    vol = _atlas_volume()
    if vol is None:
        return None
    img, data = vol
    out = np.full(data.shape, background, dtype=np.float32)
    for atlas_idx, value in values_by_atlas_idx.items():
        out[data == int(atlas_idx) + 1] = float(value)
    return nib.Nifti1Image(out, img.affine)


def partition_html(
    result,
    fname: str,
    tau_index: int,
    cut_height: float | None,
    *,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    node_size: float = 9.0,
) -> str:
    """Glass brain of the LRG partition: survivor centroids coloured by community."""
    from nilearn import plotting

    from .serializers.lrg import _lrg_entries

    _f, parts, node_ids = _lrg_entries(
        result, fname, sparsify, sparsify_alpha, sparsify_threshold
    )
    if not parts:
        return _error_page("No LRG partition available for this result.")
    tau_index = max(-len(parts), min(int(tau_index), len(parts) - 1))
    entry = parts[tau_index]
    linkage = np.asarray(entry["linkage_matrix"], dtype=float)
    n_leaves = int(linkage.shape[0] + 1)
    h = cut_height if (cut_height and cut_height > 0) else float(entry.get("flat_threshold") or 0.0)
    ids, n_clusters = leaf_cluster_ids(linkage, h)

    coords = remap.surviving_coords(result)
    if coords.shape[0] == 0:
        return _error_page("Atlas centroids are unavailable for this atlas.")
    # Leaf i -> survivor position p (filter: p = i; sparsified: p = node_ids[i]).
    labels = remap.surviving_labels(result)
    node_coords = []
    colors = []
    names = []
    ok = True
    for i in range(n_leaves):
        p = node_ids[i] if node_ids is not None else i
        if not (0 <= p < coords.shape[0]):
            ok = False
            break
        node_coords.append(coords[p])
        colors.append(LRG_PALETTE[int(ids[i]) % len(LRG_PALETTE)])
        nm = labels[p]["short"] if 0 <= p < len(labels) else f"node-{p}"
        names.append(f"{nm} · c{int(ids[i])}")
    if not ok or not node_coords:
        return _error_page("Could not align the partition to atlas centroids.")

    view = plotting.view_markers(
        np.asarray(node_coords), marker_color=colors, marker_labels=names,
        marker_size=node_size * 1.3,
    )
    return view.get_standalone()


def scalar_filled_html(
    values_by_atlas_idx: dict[int, float],
    *,
    symmetric: bool = True,
    vmax: float | None = None,
    title: str | None = None,
) -> str:
    """Glass brain of a continuous per-ROI scalar painted on the parcellation."""
    from nilearn import plotting

    if not values_by_atlas_idx:
        return _error_page("No per-ROI values to display.")
    roi_img = _roi_image(values_by_atlas_idx)
    if roi_img is None:
        return _error_page("Atlas parcellation volume is unavailable.")
    vals = np.array(list(values_by_atlas_idx.values()), dtype=float)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    vmax = vmax or 1.0
    cmap = "RdBu_r" if symmetric else "magma"
    view = plotting.view_img(
        roi_img,
        bg_img=False,
        cmap=cmap,
        symmetric_cmap=symmetric,
        vmax=vmax,
        vmin=-vmax if symmetric else 0.0,
        threshold=None,
        colorbar=True,
        title=title,
    )
    return view.get_standalone()


@lru_cache(maxsize=64)
def _partition_cached(
    dataset: str, label: str, fname: str, tau_index: int, cut_key: float,
    sparsify: str, alpha: float, threshold: float, node_size: float, _mtime: float,
) -> str:
    from .catalog import dataset_dir
    from .loaders import get_results

    rc = get_results(dataset_dir(dataset))
    return partition_html(
        rc[label], fname, tau_index, None if cut_key <= 0 else cut_key,
        sparsify=None if sparsify == "filter" else sparsify,
        sparsify_alpha=alpha, sparsify_threshold=threshold, node_size=node_size,
    )


def render_partition(
    dataset: str,
    label: str,
    filter_name: str | None,
    tau_index: int = -1,
    cut_height: float | None = None,
    *,
    sparsify: str | None = None,
    sparsify_alpha: float = 0.05,
    sparsify_threshold: float = 0.3,
    node_size: float = 9.0,
) -> str:
    """Cached standalone HTML for the LRG-partition glass brain (or an error page)."""
    from .catalog import dataset_dir
    from .loaders import get_results
    from .serializers.network import resolve_filter

    directory = dataset_dir(dataset)
    if directory is None:
        return _error_page(f"Unknown dataset {dataset!r}.")
    rc = get_results(directory)
    try:
        result = rc[label]
    except KeyError:
        return _error_page(f"Unknown result {label!r}.")
    # resolve_filter over LRG filters (partition brain follows the LRG hierarchy).
    filters = list(result.lrg_results.keys()) if result.lrg_results else []
    fname = filter_name if filter_name in filters else (filters[0] if filters else None)
    if fname is None:
        fname = resolve_filter(result, filter_name)
    if fname is None:
        return _error_page("No LRG hierarchy available for this result.")
    mtime = (directory / "results.pkl").stat().st_mtime
    # Pass the cut at FULL precision — the optimal cut sits exactly on a merge
    # height, so rounding it down would drop that merge and add a spurious cluster
    # (making the brain disagree with the dendrogram/clustered-network). The float
    # is a stable cache key: the same slider position yields the same value.
    return _partition_cached(
        dataset, label, fname, int(tau_index),
        float(cut_height) if cut_height else 0.0,
        sparsify or "filter", round(float(sparsify_alpha), 4),
        round(float(sparsify_threshold), 4), round(float(node_size), 1), mtime,
    )
