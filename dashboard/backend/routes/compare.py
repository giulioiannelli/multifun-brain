"""Comparison routes: two-selection dendrogram/hierarchy comparison.

``GET /api/compare`` returns JSON (Baker's gamma, cophenetic r, per-ROI shift,
observed-vs-null histogram data, top reorganised/stable). ``GET /api/compare/brain``
returns a standalone nilearn glass-brain page (per-ROI calibrated shift painted on
the parcellation) for an ``<iframe>``.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ..encode import clean
from ..serializers import compare as compare_mod

router = APIRouter()


@router.get("/compare")
def compare(
    dataset_a: str,
    label_a: str,
    dataset_b: str,
    label_b: str,
    filter_a: str | None = Query(default=None),
    filter_b: str | None = Query(default=None),
    null_kind: str = Query(default="none"),
    n_surrogates: int = Query(default=60),
    backbone: str = Query(default="lans"),
) -> dict:
    """Compare two results' LRG hierarchies."""
    spec = compare_mod.compare_spec(
        dataset_a, label_a, filter_a, dataset_b, label_b, filter_b,
        null_kind=null_kind, n_surrogates=n_surrogates, backbone=backbone,
    )
    return clean(spec)


@router.get("/compare/brain", response_class=HTMLResponse)
def compare_brain(
    dataset_a: str,
    label_a: str,
    dataset_b: str,
    label_b: str,
    filter_a: str | None = Query(default=None),
    filter_b: str | None = Query(default=None),
    null_kind: str = Query(default="none"),
    n_surrogates: int = Query(default=60),
    backbone: str = Query(default="lans"),
) -> HTMLResponse:
    """Glass brain of the per-ROI comparison scalar (calibrated cophenetic shift)."""
    html = compare_mod.render_compare_brain(
        dataset_a, label_a, filter_a, dataset_b, label_b, filter_b,
        null_kind=null_kind, n_surrogates=n_surrogates, backbone=backbone,
    )
    return HTMLResponse(content=html)
