# multifun-brain dashboard

Interactive browser GUI for exploring brain-network analysis results. A FastAPI
backend reuses the `multifunbrain` library (loads result bundles, serialises them
to JSON plot specs) and serves a React + Plotly frontend from a single port — so
collaborators open **one localhost URL**, no code required.

## Quick start

```bash
# from the repo root, with the multifunbrain env active
pip install -e ".[dashboard]"     # FastAPI + uvicorn (one-time)
./dashboard/run.sh                # builds the frontend, serves at http://localhost:8000
```

First run installs the frontend's npm dependencies (Node ≥ 18 required) and
builds it; subsequent runs are fast.

## Development (hot reload)

```bash
# terminal 1 — backend with autoreload
python -m uvicorn dashboard.backend.app:app --reload --port 8000

# terminal 2 — Vite dev server (proxies /api to :8000)
cd dashboard/frontend && npm run dev      # http://localhost:5173
```

## What it reads

The catalog scans `MFB_RESULTS_ROOT` (default
`data/correlation_matrices_results/`) for any directory containing a
`results.pkl` and exposes each as a selectable dataset. Override roots with
env vars: `MFB_RESULTS_ROOT`, `MFB_DATA_ROOT`, `MFB_ATLAS_DIR`,
`MFB_DASHBOARD_CACHE`, `MFB_DASHBOARD_PORT`.

## Layout

| Path | Purpose |
|---|---|
| `backend/app.py` | FastAPI app; mounts `/api` + the built frontend |
| `backend/catalog.py` | dataset/label discovery + facet parsing |
| `backend/loaders.py` | memoised `load_results` (keyed by mtime) |
| `backend/atlas.py` · `remap.py` | Schaefer labels/centroids + dead-region re-indexing |
| `backend/serializers/` | `PipelineResult` → JSON plot specs (one module per section) |
| `backend/routes/` | `/api` endpoints |
| `frontend/src/` | React + Plotly UI (selector, views, plot components) |

## Status

Phase A (vertical slice): catalog + interactive correlation heatmap with atlas
hover. Subsequent phases add the remaining plots, the 3-D brain, comparison
views, and the folder-ingestion workflow. See the project plan for the roadmap.
