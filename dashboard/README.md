# multifun-brain dashboard

Interactive browser GUI for exploring brain-network analysis results. A FastAPI
backend reuses the `multifunbrain` library (loads result bundles, serialises them
to JSON plot specs) and serves a React + Plotly frontend from a single port — so
collaborators open **one localhost URL**, no code required.

---

> **Who needs to install what?** Node.js is needed **only to build the frontend
> once** — serving it afterwards is pure Python. So there are two models:
>
> - **One person hosts, everyone else just opens a URL** (best for non-coders).
>   Only the host machine needs anything installed; collaborators need *only a
>   browser*. See [§3.1](#31-let-collaborators-just-open-a-url-zero-install).
> - **Each person runs it themselves.** They need the environment below — and
>   the **conda environment already bundles Node**, so nobody installs Node by
>   hand.

## 1. Prerequisites (one-time, on the machine that *runs* the dashboard)

The recommended way gets Python **and** Node in one step — Node ships inside the
conda environment, so collaborators never install it separately:

```bash
conda env create -f multifun-brain.yml     # includes nodejs + the dashboard deps
conda activate multifun-brain
```

<details>
<summary>Already have a Python env and don't use conda?</summary>

Then you need **Node.js ≥ 18** on `PATH` (e.g. via your system package manager,
[nvm](https://github.com/nvm-sh/nvm), or `conda install -c conda-forge nodejs`),
plus the dashboard's Python deps:

```bash
pip install -e ".[dashboard]"     # FastAPI + uvicorn
```
</details>

## 2. Setup

Nothing extra — `conda env create` above already ran `pip install -e
".[dev,viz,dashboard]"`. (With the manual route, `pip install -e ".[dashboard]"`
is the only step.)

## 3. Run it

```bash
./dashboard/run.sh                # builds the frontend (first run only), then serves
```

Then open **http://localhost:8000** in any browser. The first launch runs
`npm install` + a build (needs Node); every later launch skips straight to
serving (no Node needed). `Ctrl-C` stops the server.

### 3.1 Let collaborators "just open a URL" (zero install)

To run the dashboard on one machine (e.g. your work PC) and let collaborators on
the **same network** reach it without installing *anything*, bind to all
interfaces with `MFB_DASHBOARD_HOST=0.0.0.0`:

```bash
MFB_DASHBOARD_HOST=0.0.0.0 ./dashboard/run.sh
```

On startup it prints the LAN URL to share, e.g.:

```
[dashboard] on this network, others can open:  http://192.168.1.42:8000
```

Collaborators open that `http://<host-ip>:8000` in their browser — no Python, no
Node, nothing to install on their side. Notes:

- The host machine still needs the environment (it builds + serves); `run.sh`
  handles the one-time build.
- Find the host IP manually with `hostname -I` (Linux) if the printed line is
  empty.
- If it's unreachable, allow **inbound TCP on the port** (default `8000`) through
  the host's firewall.
- Collaborators must be on the **same LAN/VPN** as the host. Reaching it from
  outside the network is the "hostable later" step (reverse proxy / deploy), not
  covered here.

Useful knobs:

```bash
MFB_DASHBOARD_PORT=8001 ./dashboard/run.sh    # serve on a different port
MFB_DASHBOARD_REBUILD=1 ./dashboard/run.sh    # force a frontend rebuild
```

> **"The page never loads."** Almost always a previous server still holding the
> port. Check `lsof -iTCP:8000 -sTCP:LISTEN`; free it with
> `kill $(lsof -tiTCP:8000 -sTCP:LISTEN)`, or just use another port (above).
> `run.sh` now refuses to start on a busy port with this same hint.

---

## 4. Where the data has to go

The dashboard reads **pre-computed result bundles** — the output of the analysis
pipeline, not raw correlation matrices. (Dropping a raw-matrix folder and having
the GUI elaborate it is a planned later phase; for now bundles are produced by
the pipeline, e.g. `scripts/april/`.)

### 4.1 Directory layout

Place each bundle under the results root — by default
`data/correlation_matrices_results/` (override with `MFB_RESULTS_ROOT`):

```
data/correlation_matrices_results/        ← MFB_RESULTS_ROOT
└── <group>/                              ← "Dataset" dropdown   (e.g. april)
    └── <category>/                       ← "Category" dropdown  (e.g. global, bands, patients)
        ├── results.pkl                   ← REQUIRED. The bundle the dashboard loads.
        ├── summary.csv                   ← optional (metrics table)
        ├── config.json                   ← optional (pipeline config used)
        └── failed_matrices.json          ← optional
```

Discovery is **recursive**: *any* directory that contains a `results.pkl`
becomes a selectable dataset. Its path relative to the root — e.g.
`april/global` — is split in the UI into a **Dataset** dropdown (`april`) and a
**Category** dropdown (`global`). So to add data you simply drop a folder
containing a `results.pkl` anywhere under the results root; it appears in the
dropdowns automatically (restart the server, or it's picked up on the next
catalog read).

### 4.2 The subject structure lives *inside* the bundle

A single `results.pkl` holds many results, each tagged with a **label** that
encodes its facets. Subjects are **not** separate folders — they are part of the
label. The label shape per aggregation level:

| Level   | Label shape                                       | Example                                   |
|---------|---------------------------------------------------|-------------------------------------------|
| global  | `global/<contrast>_<processing>`                  | `global/co2_bpfBOLD`                       |
| band    | `band/<contrast>_<processing>/<band>`             | `band/co2_bpfBOLD/s4`                      |
| patient | `patient/<subject>/<contrast>_<processing>/<band>`| `patient/sub-00246757/co2_bpfBOLD/s4`      |

- `contrast` ∈ {`co2`, `rest`}
- `processing` ∈ {`bpfBOLD`, `bpfVASO`, `MIRNoise_bold`, `optcom_bold`, `optcomMIRDenoised_bold`}
- `band` ∈ {`s4`, `s5`, `sstar`}
- `subject` — e.g. `sub-00246757`

The dashboard parses these labels into the **Result** dropdown and the
contrast / processing / band / subject filters automatically. So a *per-subject*
dataset is just a `results.pkl` whose entries use `patient/...` labels (one
bundle, many subjects) — there is nothing to rename on disk.

### 4.3 Atlas (region names on hover)

Hovering a node shows its Schaefer region name. That needs the atlas files under
`data/schaefer_2018/` (override with `MFB_ATLAS_DIR`):

```
data/schaefer_2018/
├── Schaefer2018_100Parcels_7Networks_order.txt                       ← region names + 7-network groups
└── Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz      ← parcellation volume (for 3-D, later)
```

Dead regions (NaN rows dropped during preprocessing) are remapped so hover names
stay aligned with the surviving nodes — no action needed.

### Environment overrides (summary)

| Variable | Default | Purpose |
|---|---|---|
| `MFB_RESULTS_ROOT` | `data/correlation_matrices_results/` | where result bundles are discovered |
| `MFB_DATA_ROOT` | `data/` | allowlist boundary for ingestion (future) |
| `MFB_ATLAS_DIR` | `data/schaefer_2018/` | Schaefer atlas assets |
| `MFB_DASHBOARD_CACHE` | `dashboard/.cache/` | on-disk plot-spec / centroid cache |
| `MFB_DASHBOARD_PORT` | `8000` | server port |
| `MFB_DASHBOARD_HOST` | `127.0.0.1` | bind address; set `0.0.0.0` to share on the LAN (§3.1) |

---

## 5. Development (hot reload)

```bash
# terminal 1 — backend with autoreload
python -m uvicorn dashboard.backend.app:app --reload --port 8000

# terminal 2 — Vite dev server (proxies /api to :8000)
cd dashboard/frontend && npm run dev      # http://localhost:5173
```

## 6. Code layout

| Path | Purpose |
|---|---|
| `backend/app.py` | FastAPI app; mounts `/api` + the built frontend |
| `backend/catalog.py` | dataset/label discovery + facet parsing |
| `backend/loaders.py` | memoised `load_results` (keyed by mtime) |
| `backend/atlas.py` · `remap.py` | Schaefer labels/centroids + dead-region re-indexing |
| `backend/serializers/` | `PipelineResult` → JSON plot specs (one module per section) |
| `backend/routes/` | `/api` endpoints |
| `frontend/src/` | React + Plotly UI (selector, views, plot components) |

## 7. Status

- **Done**: catalog + tabbed Explore view (descriptive · network · LRG) with the
  interactive correlation heatmap, atlas-name hover, Cytoscape 2-D network, and
  LRG dendrogram / partition-flow / Sankey.
- **Planned**: interactive 3-D brain, comparison views (CO2 vs rest, per-patient
  vs average, across variants/bands, two patients side by side), and the
  drop-a-folder ingestion workflow. See the project plan for the roadmap.
