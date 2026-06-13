"""3-D brain route: ``GET /api/brain3d`` -> standalone nilearn HTML for an iframe.

Returns ``text/html`` (not JSON) so the large self-contained nilearn page is
loaded natively by an ``<iframe src>``; errors render as a small HTML message so
the iframe always shows something.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from .. import brain3d

router = APIRouter()


@router.get("/brain3d", response_class=HTMLResponse)
def brain3d_view(
    dataset: str,
    label: str,
    filter: str | None = Query(default=None),
    mode: str = Query(default="connectome"),
    edge_quantile: float = Query(default=0.98),
) -> HTMLResponse:
    """Interactive 3-D connectome / markers view for a filtered network."""
    html = brain3d.render(dataset, label, filter, mode, edge_quantile)
    return HTMLResponse(content=html)
