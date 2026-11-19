"""Dendrogram / hierarchy comparison between two results (the Compare tab).

Contrasts the LRG hierarchies of two selections (different task / modality /
processing, at group-average or single-patient level) the way the collaborator's
``fig6`` does: Baker's gamma and cophenetic correlation on the shared ROI set, a
per-ROI cophenetic-rank *shift* (0 = same neighbourhood, 1 = fully reorganised),
and — against a strength-preserving null — the *calibrated* shift (observed −
null median; > 0 reorganised beyond chance, < 0 more stable than chance), ready to
reproject onto the brain.

Everything is keyed by **original atlas index** (not survivor position), so the
two hierarchies align ROI-to-ROI even when they dropped different dead regions,
and the per-ROI values map straight onto the parcellation for the glass brain.
"""

from __future__ import annotations

import re
from functools import lru_cache

import networkx as nx
import numpy as np

from multifunbrain.analysis.lrg.compare import (
    compare_hierarchies,
    null_shift_distribution,
)
from multifunbrain.analysis.lrg.partitions import linkage_at_tau_min

from .. import config
from ..catalog import dataset_dir
from ..graphs import derive_graph
from ..loaders import get_results
from ..remap import surviving_labels
from .lrg import resolve_filter as resolve_lrg_filter


def _atlas_graph(
    result, fname: str, backbone: str = "lans", alpha: float = 0.05
) -> tuple[nx.Graph, dict[int, str]]:
    """Backbone graph relabelled to original atlas indices, plus idx→name.

    Uses a sparse backbone (default LANS, as in the collaborator's fig6) rather
    than the dense stored filter, so the diffusion hierarchy is meaningful and the
    strength-preserving null is fast. Relabelling to atlas indices makes both
    hierarchies (and their surrogates) share one node namespace, so the
    intersection and the brain reprojection stay correct even across results with
    different dead-region drops.
    """
    graph, _eff = derive_graph(result, fname, backbone, alpha, 0.3)
    labels = surviving_labels(result)
    mapping: dict[int, int] = {}
    names: dict[int, str] = {}
    for pos in graph.nodes():
        p = int(pos)
        if 0 <= p < len(labels):
            atlas_idx = int(labels[p]["index"])
            mapping[p] = atlas_idx
            names[atlas_idx] = labels[p]["short"]
    graph_orig = nx.relabel_nodes(graph, mapping, copy=True)
    return graph_orig, names


def _is_schaefer(result) -> bool:
    return int(getattr(result, "n_regions_original", 0)) == 100


@lru_cache(maxsize=32)
def _compute_cached(
    dataset_a: str, label_a: str, fname_a: str,
    dataset_b: str, label_b: str, fname_b: str,
    null_kind: str, n_surrogates: int, backbone: str, _mtime_a: float, _mtime_b: float,
) -> dict:
    rc_a = get_results(dataset_dir(dataset_a))
    rc_b = get_results(dataset_dir(dataset_b))
    result_a, result_b = rc_a[label_a], rc_b[label_b]

    graph_a, names_a = _atlas_graph(result_a, fname_a, backbone)
    graph_b, names_b = _atlas_graph(result_b, fname_b, backbone)
    if graph_a.number_of_nodes() < 4 or graph_b.number_of_nodes() < 4:
        return {"error": "one of the networks is too small to compare"}

    try:
        Z_a, leaf_a, _ta = linkage_at_tau_min(graph_a)
        Z_b, leaf_b, _tb = linkage_at_tau_min(graph_b)
    except RuntimeError as exc:
        return {"error": f"could not build a hierarchy: {exc}"}

    null_per_atlas = None
    if null_kind == "strength" and n_surrogates > 0:
        # Surrogate caches are keyed on a per-graph label used as a filename stem.
        # Slashes in dataset/label ids would nest into missing dirs, so sanitise;
        # include the results.pkl mtime so a regenerated (materially different)
        # backbone graph doesn't silently reuse surrogates built from the old one.
        key_a = re.sub(r"[^\w.-]", "_", f"{dataset_a}|{label_a}|{fname_a}|{backbone}|{_mtime_a}")
        key_b = re.sub(r"[^\w.-]", "_", f"{dataset_b}|{label_b}|{fname_b}|{backbone}|{_mtime_b}")
        null_per_atlas = null_shift_distribution(
            graph_a, key_a, graph_b, key_b,
            n_surrogates=n_surrogates, cache_dir=config.CACHE_DIR / "compare",
        )

    cmp = compare_hierarchies(Z_a, leaf_a, Z_b, leaf_b, null_per_atlas=null_per_atlas)
    if cmp.n_common < 4:
        return {"error": f"only {cmp.n_common} ROIs in common"}

    names = {**names_b, **names_a}
    common = cmp.common.tolist()
    use_cal = cmp.has_null
    metric = cmp.calibrated if use_cal else cmp.shift

    rois = []
    for i, atlas_idx in enumerate(common):
        m = float(metric[i])
        if not np.isfinite(m):
            continue
        rois.append({
            "atlas_index": int(atlas_idx),
            "name": names.get(int(atlas_idx), f"roi-{atlas_idx}"),
            "value": m,
            "shift": float(cmp.shift[i]),
            "calibrated": float(cmp.calibrated[i]) if np.isfinite(cmp.calibrated[i]) else None,
            "null_median": float(cmp.null_median[i]) if np.isfinite(cmp.null_median[i]) else None,
        })
    rois.sort(key=lambda r: r["value"], reverse=True)

    return {
        "kind": "compare",
        "label_a": result_a.label,
        "label_b": result_b.label,
        "filter_a": fname_a,
        "filter_b": fname_b,
        "backbone": backbone,
        "bakers_gamma": cmp.bakers_gamma,
        "cophenetic_r": cmp.cophenetic_r,
        "n_common": cmp.n_common,
        "null_kind": null_kind if use_cal else "none",
        "metric": "calibrated" if use_cal else "shift",
        "rois": rois,
        "observed": cmp.shift.tolist(),
        "pooled_null": cmp.pooled_null.tolist(),
        "vmax_abs": float(np.nanmax(np.abs(metric))) if metric.size else 1.0,
        "brain_available": _is_schaefer(result_a) and _is_schaefer(result_b),
    }


def _resolve(dataset: str, label: str, filter_name: str | None):
    directory = dataset_dir(dataset)
    if directory is None:
        return None, None, None, None
    rc = get_results(directory)
    try:
        result = rc[label]
    except KeyError:
        return None, None, None, None
    fname = resolve_lrg_filter(result, filter_name)
    mtime = (directory / "results.pkl").stat().st_mtime
    return result, fname, mtime, directory


def compare_spec(
    dataset_a: str, label_a: str, filter_a: str | None,
    dataset_b: str, label_b: str, filter_b: str | None,
    *, null_kind: str = "none", n_surrogates: int = 60, backbone: str = "lans",
) -> dict:
    """JSON comparison of two results' LRG hierarchies (see module docstring)."""
    ra, fa, ma, _ = _resolve(dataset_a, label_a, filter_a)
    rb, fb, mb, _ = _resolve(dataset_b, label_b, filter_b)
    if ra is None or rb is None:
        return {"kind": "compare", "error": "unknown result selection"}
    if fa is None or fb is None:
        return {"kind": "compare", "error": "one result has no LRG hierarchy"}
    kind = "strength" if null_kind == "strength" else "none"
    return _compute_cached(
        dataset_a, label_a, fa, dataset_b, label_b, fb,
        kind, int(n_surrogates), backbone or "lans", ma, mb,
    )


def comparison_roi_values(
    dataset_a: str, label_a: str, filter_a: str | None,
    dataset_b: str, label_b: str, filter_b: str | None,
    *, null_kind: str = "none", n_surrogates: int = 60, backbone: str = "lans",
) -> tuple[dict[int, float], float, bool]:
    """``({atlas_idx: value}, vmax_abs, symmetric)`` for the glass-brain reprojection."""
    spec = compare_spec(
        dataset_a, label_a, filter_a, dataset_b, label_b, filter_b,
        null_kind=null_kind, n_surrogates=n_surrogates, backbone=backbone,
    )
    if spec.get("error") or not spec.get("rois"):
        return {}, 1.0, True
    values = {int(r["atlas_index"]): float(r["value"]) for r in spec["rois"]}
    symmetric = spec.get("metric") == "calibrated"
    return values, float(spec.get("vmax_abs", 1.0)), symmetric


@lru_cache(maxsize=16)
def _brain_cached(
    dataset_a: str, label_a: str, filter_a: str,
    dataset_b: str, label_b: str, filter_b: str,
    null_kind: str, n_surrogates: int, backbone: str, _mtime_a: float, _mtime_b: float,
) -> str:
    from ..glassbrain import _error_page, scalar_filled_html

    values, vmax, symmetric = comparison_roi_values(
        dataset_a, label_a, filter_a, dataset_b, label_b, filter_b,
        null_kind=null_kind, n_surrogates=n_surrogates, backbone=backbone,
    )
    if not values:
        return _error_page("No comparison values to reproject (atlas mismatch or too few ROIs).")
    title = "calibrated shift · +reorganised / −stable" if symmetric else "cophenetic shift"
    return scalar_filled_html(values, symmetric=symmetric, vmax=vmax, title=title)


def render_compare_brain(
    dataset_a: str, label_a: str, filter_a: str | None,
    dataset_b: str, label_b: str, filter_b: str | None,
    *, null_kind: str = "none", n_surrogates: int = 60, backbone: str = "lans",
) -> str:
    """Cached glass-brain HTML of the per-ROI comparison scalar (or an error page)."""
    from ..glassbrain import _error_page

    ra, fa, ma, _ = _resolve(dataset_a, label_a, filter_a)
    rb, fb, mb, _ = _resolve(dataset_b, label_b, filter_b)
    if ra is None or rb is None or fa is None or fb is None:
        return _error_page("Unknown result selection for the comparison brain.")
    kind = "strength" if null_kind == "strength" else "none"
    return _brain_cached(
        dataset_a, label_a, fa, dataset_b, label_b, fb,
        kind, int(n_surrogates), backbone or "lans", ma, mb,
    )
