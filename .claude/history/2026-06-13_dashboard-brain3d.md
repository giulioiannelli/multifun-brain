# 2026-06-13 — Dashboard Phase B′: interactive 3-D brain

Branch: `dashboard`. Adds a **Brain 3-D** tab (between Network and LRG) that
renders a filtered network as a rotatable nilearn view.

## What changed

- **New `dashboard/backend/brain3d.py`** — builds the standalone nilearn HTML:
  - `connectome` — `view_connectome(adjacency, centroids, edge_threshold="{q}%")`
    over the thresholded filtered-network adjacency; nodes coloured by 7-network.
  - `markers` — `view_markers(centroids, marker_color=…, marker_labels=…)`.
  - Node identity flows through `remap.surviving_coords` / `surviving_labels`
    (graph node IDs index the survivor-aligned arrays, same as `network_spec`);
    centroids come from `atlas.parcel_centroids()` (cached). nilearn imported
    lazily. HTML cached by `(dataset,label,filter,mode,edge_quantile,mtime)`.
- **New `dashboard/backend/routes/brain3d.py`** — `GET /api/brain3d` returns
  `text/html` (HTMLResponse), so the ~2.5 MB self-contained page loads natively
  in an `<iframe>` instead of round-tripping as JSON. Errors render as a small
  HTML message so the iframe always shows something. Wired into `app.py`.
- **Frontend**: `App` TABS gains `Brain 3-D`; `ExploreTab` adds it. New
  `components/plots/Brain3D.tsx` builds the `/api/brain3d?...` URL and embeds an
  `<iframe>` (remounts via `key` on param change). `ExploreView` shows the filter
  selector + a **View** toggle (Connectome / Markers) + an **Edges** quantile
  select (top 10/5/2/1%) for connectome. `.brain3d-frame` CSS (centred, capped
  width, 640 px).

## Verification

- `ruff check dashboard/backend` clean; `npm run build` clean.
- Backend TestClient: `/api/brain3d` returns 200 `text/html` for both modes;
  bad label → error page (still 200 HTML).
- Headless-Chrome e2e: tab order `Signal|Correlation|Network|Brain 3-D|LRG`;
  controls Filter/View/Edges present; iframe loads the nilearn "Connectome plot"
  page (1 plotly div). With software WebGL (SwiftShader) the **connectome**
  (glass brain + red edges + weight colourbar) and **markers** (parcels coloured
  by 7-network, atlas labels) both render with no JS errors. (Plain headless
  `--disable-gpu` shows nilearn's "WebGL not supported" notice — a headless
  limitation, not an app issue; real browsers render fine.)

## Notes

- nilearn is already a core dep; no new dependency.
- view_connectome/markers produce a fixed-width figure, so the iframe is centred
  and width-capped to avoid a wide empty margin.