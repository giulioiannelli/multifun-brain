"""Catalog + per-result metadata routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..catalog import dataset_dir, list_datasets, parse_label
from ..loaders import get_results

router = APIRouter()


@router.get("/catalog")
def catalog() -> dict:
    """All discoverable datasets, their items, and aggregated facets."""
    return {"datasets": list_datasets()}


@router.get("/result")
def result_meta(dataset: str, label: str) -> dict:
    """Metadata for one result (region counts, available filters, config, error)."""
    directory = dataset_dir(dataset)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset!r}")
    rc = get_results(directory)
    try:
        r = rc[label]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown label {label!r}") from exc

    return {
        "dataset": dataset,
        "label": r.label,
        **parse_label(r.label or ""),
        "n_regions_original": int(r.n_regions_original),
        "n_dropped": int(len(r.dropped_regions)),
        "filters": list(r.network_analyses.keys()),
        "lrg_filters": list(r.lrg_results.keys()),
        "has_corr": r.corr_prepared is not None,
        "error": r.error,
    }
