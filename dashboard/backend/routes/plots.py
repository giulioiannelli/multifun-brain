"""Plot-spec routes: ``GET /api/plot/{kind}`` -> JSON for interactive rendering.

Shared query params (``filter``, ``tau_index``, ``edge_quantile``) are forwarded
to every serializer as keyword args; serializers ignore the ones they don't use.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..catalog import dataset_dir
from ..loaders import get_results
from ..serializers import PLOT_KINDS

router = APIRouter()


@router.get("/plot/kinds")
def plot_kinds() -> dict:
    """Available plot kinds (for frontend discovery)."""
    return {"kinds": sorted(PLOT_KINDS)}


@router.get("/plot/{kind}")
def plot(
    kind: str,
    dataset: str,
    label: str,
    filter: str | None = Query(default=None),
    tau_index: int = Query(default=-1),
    edge_quantile: float = Query(default=0.9),
) -> dict:
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
    return serializer(
        r, filter_name=filter, tau_index=tau_index, edge_quantile=edge_quantile
    )
