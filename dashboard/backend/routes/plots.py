"""Plot-spec routes: ``GET /api/plot/{kind}`` -> JSON for interactive rendering."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..catalog import dataset_dir
from ..loaders import get_results
from ..serializers import PLOT_KINDS

router = APIRouter()


@router.get("/plot/kinds")
def plot_kinds() -> dict:
    """Available plot kinds (for frontend discovery)."""
    return {"kinds": sorted(PLOT_KINDS)}


@router.get("/plot/{kind}")
def plot(kind: str, dataset: str, label: str) -> dict:
    """Serialise *kind* for the result identified by (*dataset*, *label*)."""
    serializer = PLOT_KINDS.get(kind)
    if serializer is None:
        raise HTTPException(status_code=404, detail=f"unknown plot kind {kind!r}")
    directory = dataset_dir(dataset)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset!r}")
    rc = get_results(directory)
    try:
        r = rc[label]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown label {label!r}") from exc
    return serializer(r)
