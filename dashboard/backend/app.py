"""FastAPI application: mounts ``/api`` routers and serves the built frontend.

Run (dev):   uvicorn dashboard.backend.app:app --reload --port 8000
Run (local): ./dashboard/run.sh   (builds the frontend, then serves it here)

In dev the Vite server proxies ``/api`` to this process; in local/prod the built
React bundle is served from the same port, so collaborators open one URL.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .routes import brain3d as brain3d_routes
from .routes import catalog as catalog_routes
from .routes import compare as compare_routes
from .routes import plots as plots_routes
from .routes import signal as signal_routes

app = FastAPI(title="multifun-brain dashboard", version="0.1.0")

# Local tool: permissive CORS so the Vite dev server (:5173) can call :8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cache_control(request: Request, call_next):
    """Stop the browser serving a stale bundle after a rebuild.

    Hashed assets (``/assets/<name>-<hash>.js``) are immutable, so cache them
    forever; ``index.html`` and API responses must always be revalidated so a
    reload always loads the current build and fresh data.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif "/assets/" in path:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response

app.include_router(catalog_routes.router, prefix="/api")
app.include_router(plots_routes.router, prefix="/api")
app.include_router(signal_routes.router, prefix="/api")
app.include_router(brain3d_routes.router, prefix="/api")
app.include_router(compare_routes.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "results_root": str(config.RESULTS_ROOT)}


# Serve the built SPA last so it only catches non-/api paths.
if config.FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(config.FRONTEND_DIST), html=True),
        name="frontend",
    )
