# Dashboard — how we build it

The dashboard (`dashboard/`) is **the presentation layer** of the project:
collaborators and the PI look at results there, not in notebooks. Every new
plot, comparison or exploration view lands as a dashboard panel. This page is
the agent-facing contract; the collaborator-facing reference (install, data
layout, env vars) is `dashboard/README.md` + `SETUP.md`.

## Architecture in one picture

```
results.pkl ──► backend/loaders.get_results (lru_cache keyed by mtime)
                   │
                   ▼
   backend/serializers/<section>.py   PipelineResult ──► JSON "plot spec" (dict)
                   │                  registered in serializers/__init__.PLOT_KINDS
                   ▼
   backend/routes/<area>.py           GET /api/plot/{kind}?dataset=&label=&…
                   │
                   ▼
   frontend/src/hooks/usePlot.ts      fetch + cancel stale responses
                   │
                   ▼
   components/PlotPanel.tsx           dispatch spec by kind ──►
        components/plots/figures.ts   buildX(spec, opts) → Plotly figure
        components/plots/NetworkGraph Cytoscape 2-D network
        components/plots/Brain3D …    nilearn 3-D brain HTML in an <iframe>
```

| Path | Role |
|---|---|
| `backend/app.py` | FastAPI app; mounts `/api` and serves the built frontend |
| `backend/catalog.py`, `facets.py` | discover bundles under `MFB_RESULTS_ROOT`; parse labels into facets |
| `backend/remap.py`, `atlas.py` | survivor ↔ atlas index mapping, region names, centroids |
| `backend/graphs.py` | `derive_graph` — the one place a result + sparsify choice becomes a graph |
| `backend/serializers/` | one module per section: `descriptive`, `network`, `lrg`, `signal`, `compare` |
| `backend/routes/` | `/api` endpoints: `catalog`, `plots`, `signal`, `brain3d`, `compare` |
| `backend/timeseries.py` | raw `.ts.1D` reader + EMD bands for the Signal tab |
| `backend/glassbrain.py`, `brain3d.py` | nilearn interactive brains (connectome / markers / partitions) |
| `frontend/src/App.tsx` | tab bar: Pipeline · Signal · Correlation · Network · Brain 3-D · LRG · Compare |
| `frontend/src/views/` | one view per tab (`ExploreView` hosts Correlation/Network/Brain 3-D) |
| `frontend/src/components/SelectorBar.tsx` | cascading dataset → facet → result selector |
| `test/test_dashboard_*.py`, `test/test_lrg_dashboard.py` | backend tests (serializers, graphs, catalog, signal) |

## Rules

1. **No analysis logic in the dashboard.** Serializers call `multifunbrain`
   and reshape its output into JSON. If a computation is reusable (a new
   metric, a comparison, a layout), it goes into the library first
   (e.g. `analysis/lrg/compare.py`, `analysis/lrg/layout.py`) and the
   serializer imports it. Dashboard-only glue (caching, JSON shaping,
   colour assignment) stays in `dashboard/backend/`.
2. **No matplotlib in the dashboard.** Everything is interactive: Plotly
   (heatmaps, curves, distributions, Sankey), Cytoscape.js (2-D networks),
   nilearn `view_connectome` / `view_markers` HTML (3-D brain).
3. **Keep the nilearn 3-D brain.** The rotatable translucent grey brain served
   as standalone HTML in an `<iframe>` is what the user wants. A Plotly
   `scatter3d` rewrite and a server-rendered PNG were both tried and rejected
   (no brain mesh, WebGL breakage). See `never-always/never.md`.
4. **Node identity.** Graph node ids are *survivor positions* (dead regions
   already dropped). `remap.surviving_labels(result)[pos]` gives the name and
   the original atlas `index`; `remap.surviving_coords(result)` gives MNI
   centroids. Every graph returned by `graphs.derive_graph` keeps this
   indexing, so hover names and 3-D coordinates stay aligned across tabs.
5. **One sparsification path.** Network, LRG and Brain 3-D all thin the
   graph through `graphs.derive_graph`. Never re-derive a backbone inline in a
   serializer.
6. **Spec shape.** A serializer is `(result, **params) -> dict`, swallows the
   params it doesn't use via `**_`, always returns `"kind"` and `"label"`, and
   reports a recoverable problem as `{"error": "…"}` (rendered in the card)
   rather than raising. Pass the dict through `encode.clean` so numpy types
   become JSON.
7. **Selectors cascade.** Facet dropdowns only offer combinations that exist
   in the loaded bundle (contrast → processing → band → subject). New
   selectors follow the same pattern; never show an option that 404s.
8. **The Signal tab keys files differently** — `contrast` = task (co2/rest),
   `processing` = the `desc-` token — from the result tabs' `facets.parse_variant`
   (task · contrast · processing). Don't mix the two parsers.

## Recipes

### Add a plot to an existing tab

1. Serializer: add `xyz_spec(result, *, …, **_) -> dict` in the matching
   `backend/serializers/<section>.py`; register `"xyz": …` in
   `serializers/__init__.PLOT_KINDS`.
2. New query parameter? Add it to the explicit signature of
   `routes/plots.py::plot` **and** forward it — the route passes named kwargs,
   so an unlisted param silently never reaches the serializer.
3. Frontend: write `buildXyz(spec, opts)` in `components/plots/figures.ts`
   and register it in `PlotPanel.tsx::FIGURE_BUILDERS` (or a dedicated
   component for non-Plotly renderers).
4. Place `<PlotPanel kind="xyz" title="…" params={…} />` in the view.
5. Test the serializer on a synthetic `PipelineResult` in
   `test/test_dashboard_<area>.py`.

### Add a tab

Add the name to `TABS` in `App.tsx`, create `views/<Name>View.tsx`, and add a
route module under `backend/routes/` if the tab needs endpoints that are not
`/api/plot/{kind}` (register the router in `app.py`).

### Gotchas already paid for

- `/api/plot` defaults `edge_quantile=0.9`, which drops ~90% of edges. Views
  that need the full graph (e.g. `lrg_network`) must send `edge_quantile=0`.
- A dendrogram cut height must travel at **full float precision**: the optimal
  cut sits exactly on a merge height, and rounding adds a phantom community.
- Disconnected sparsified graphs: `diffusion_distance_matrix` maps infinite
  cross-component distance to a large finite value so MDS doesn't collapse.
- Disk caches (null surrogates, centroids) must key on `results.pkl` mtime and
  any RNG seed, or they go stale.
- HarvardOxford-48 parcels are bilateral: their centroids sit near the
  mid-line. That is faithful, not a bug.

## Verify a change (before saying it's done)

Tests alone are not enough; the user judges the dashboard by looking at it.

```bash
pytest test/ -q                                   # includes test_dashboard_*
ruff check multifunbrain/ dashboard/backend/
cd dashboard/frontend && npx tsc --noEmit && npm run build && cd -
# serve on a free port (8000/8001 are often taken by the user's own instance)
python -m uvicorn dashboard.backend.app:app --port 8011
curl -s "localhost:8011/api/plot/<kind>?dataset=<ds>&label=<label>" | head -c 400
```

Then open the tab in the browser (Claude-in-Chrome) and look at it: screenshot
the changed panel, exercise the new control, check the console for errors.
Report what you saw. If you could not open a browser, say so explicitly.

Data lives in the main checkout's `data/` (gitignored); a worktree has an
empty `data/`, so point `MFB_RESULTS_ROOT`, `MFB_RAW_DATA_ROOT` and
`MFB_ATLAS_DIR` at the main checkout when serving from a worktree.
