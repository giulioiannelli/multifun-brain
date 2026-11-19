"""3-D brain route: ``GET /api/brain3d`` -> standalone nilearn HTML for an iframe.

Returns ``text/html`` (not JSON) so the large self-contained nilearn page is
loaded natively by an ``<iframe src>``; errors render as a small HTML message so
the iframe always shows something.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from .. import brain3d, glassbrain

router = APIRouter()


@router.get("/brain3d", response_class=HTMLResponse)
def brain3d_view(
    dataset: str,
    label: str,
    filter: str | None = Query(default=None),
    mode: str = Query(default="connectome"),
    edge_quantile: float = Query(default=0.98),
    node_size: float = Query(default=9.0),
    sparsify: str | None = Query(default=None),
    sparsify_alpha: float = Query(default=0.05),
    sparsify_threshold: float = Query(default=0.3),
) -> HTMLResponse:
    """Interactive 3-D connectome / markers view for a (optionally sparsified) network."""
    html = brain3d.render(
        dataset, label, filter, mode, edge_quantile, node_size,
        sparsify, sparsify_alpha, sparsify_threshold,
    )
    return HTMLResponse(content=html)


@router.get("/lrg_brain", response_class=HTMLResponse)
def lrg_brain_view(
    dataset: str,
    label: str,
    filter: str | None = Query(default=None),
    tau_index: int = Query(default=-1),
    cut_height: float | None = Query(default=None),
    node_size: float = Query(default=9.0),
    sparsify: str | None = Query(default=None),
    sparsify_alpha: float = Query(default=0.05),
    sparsify_threshold: float = Query(default=0.3),
) -> HTMLResponse:
    """Glass brain of the LRG partition (parcels coloured by community at τ / cut)."""
    html = glassbrain.render_partition(
        dataset, label, filter, tau_index, cut_height,
        sparsify=sparsify, sparsify_alpha=sparsify_alpha,
        sparsify_threshold=sparsify_threshold, node_size=node_size,
    )
    return HTMLResponse(content=html)
