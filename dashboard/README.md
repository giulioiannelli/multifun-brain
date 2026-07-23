# multifun-brain dashboard

Interactive browser GUI for exploring brain-network analysis results. A FastAPI
backend reuses the `multifunbrain` library (loads result bundles, serialises them
to JSON plot specs) and serves a React + Plotly frontend from a single port — so
collaborators open **one localhost URL**, no code required.

> **Just getting started?** Don't read this whole file — follow
> **[`../SETUP.md`](../SETUP.md)**, the cross-platform clone→browser guide. The
> short version, from a fresh clone on Windows/macOS/Linux:
> ```bash
> conda env create -f multifun-brain.yml && conda activate multifun-brain
> python quickstart.py            # auto-detects data, caches results, builds, serves
> ```
> `python quickstart.py serve` is the everyday launch (after `git pull`). The rest
> of this file is the **deep reference** — data locations, env overrides, the
> result-bundle label scheme, and the dev/hot-reload workflow.

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
pipeline. You normally never build these by hand: `python quickstart.py` detects
each `raw_data_<id>` folder and **computes + caches** its bundles for you the
first time (raw timecourses → EMD bands → correlation → pipeline; see
`scripts/raw_ingest/`). This section documents the on-disk layout that ingest
produces (and that a maintainer can also ship pre-computed), for when you need to
know where things live.

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
└── Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz      ← parcellation volume (parcel centroids for the 3-D brain)
```

Dead regions (NaN rows dropped during preprocessing) are remapped so hover names
stay aligned with the surviving nodes — no action needed. The **Brain 3-D** and
**LRG glass-brain** tabs render nilearn's **interactive 3-D brain** (the rotatable
translucent grey brain with coloured nodes and, for connectomes, edges), served as
a self-contained page inside an `<iframe>` — drag to rotate, hover a node for its
name. It needs a WebGL-capable browser. Parcel centroids are derived once and
cached under `MFB_DASHBOARD_CACHE` for both atlases: Schaefer-100 (from the order
NIfTI) and HarvardOxford-48 (from nilearn's bundled cortical atlas), so the brain
renders for the HarvardOxford set too — its 48 parcels are bilateral, so their
centroids sit near the mid-line.

### 4.4 Raw timecourses (the **Signal** tab)

The Signal tab plots the *raw* ROI timecourses (before any correlation), so it
needs the AFNI `.ts.1D` files under a raw-data root (override with
`MFB_RAW_DATA_ROOT`), one folder per subject:

```
data/raw_data_schaefer100_april2026/      ← MFB_RAW_DATA_ROOT (default)
├── sub-00246757/
│   ├── ..._task-co2_run-02_..._desc-bpfBOLD.ts.1D
│   ├── ..._task-rest_run-02_..._desc-optcom_bold.ts.1D
│   └── ...                               ← one .ts.1D per (task, processing)
└── sub-.../
```

Each file is `3dROIstats` output: `n_timepoints` rows × 100 region columns. The
filename's `task-{co2|rest}` and `desc-{variant}` drive the Subject / Contrast /
Processing selectors automatically; nothing else to configure. If the folder is
absent the Signal tab simply says so and the other tabs still work.

**Raw-dataset naming.** Raw sets follow `raw_data_<atlas><batch>` so they're
identifiable at a glance. Three exist:

| Directory | Atlas | Filename scheme | Signal tab |
|---|---|---|---|
| `raw_data_schaefer100_april2026` | Schaefer 100 | BIDS `sub-/ses-/task-/desc-` | ✅ full (carpet · channel · EMD · cohort/band) |
| `raw_data_schaefer100_november2025` | Schaefer 100 | AFNI `kw…` | ✅ carpet · channel · EMD |
| `raw_data_harvardoxford48` | HarvardOxford 48 | AFNI `kw…` | ✅ carpet · channel · EMD |

All three are selectable from the **Dataset** dropdown in the Signal tab. The
reader handles both filename schemes (BIDS, and the older ``kw…``: subject from
the parent folder, modality token as the processing facet, no co2/rest; empty
``*4D`` placeholder files and any ``discarded/`` subtree are skipped). The older
two have no contrast and no known TR, so the **cohort histogram / per-band
reconstruction** (which need a contrast + Hz) stay April-only. Channel names come
from each set's atlas: the Schaefer order file for the Schaefer-100 sets, and the
**HarvardOxford cortical** FSL label XML (``data/HarvardOxford-Cortical.xml``,
override ``MFB_HARVARD_OXFORD_XML``) for the 48-parcel set — so a channel reads
e.g. *"Heschl's Gyrus"* rather than *region-44*. If that XML is absent the labels
fall back to generic ``region-N``. `MFB_RAW_DATA_ROOT` sets which dataset is the
default.

**EMD frequency bands.** The cohort histogram and per-band reconstruction follow
the collaborator's notebook: each IMF's characteristic frequency is its **median
instantaneous frequency** (HHT) in **Hz**, using the per-variant TR (`bpf*`
1.353 s, `optcom/MIR*` 0.98 s → `fs = 1/TR`). IMFs are classified into the fixed
**canonical** slow-oscillation bands (Slow-5 `0.010–0.027`, Slow-4
`0.027–0.073`, S\* `0.073–0.180` Hz). A **Bands** toggle switches to
**data-driven** edges (geometric midpoints between the per-IMF clusters) for
comparison — on the April cohort the two nearly coincide.

### Environment overrides (summary)

| Variable | Default | Purpose |
|---|---|---|
| `MFB_RESULTS_ROOT` | `data/correlation_matrices_results/` | where result bundles are discovered |
| `MFB_RAW_DATA_ROOT` | `data/raw_data_schaefer100_april2026/` | raw `.ts.1D` timecourses for the Signal tab |
| `MFB_DATA_ROOT` | `data/` | allowlist boundary for ingestion (future) |
| `MFB_ATLAS_DIR` | `data/schaefer_2018/` | Schaefer atlas assets |
| `MFB_HARVARD_OXFORD_XML` | `data/HarvardOxford-Cortical.xml` | FSL labels → channel names for the HarvardOxford-48 raw set |
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
- **Done**: the **Pipeline** tab — a static, data-free methodology scheme
  (boxes-and-arrows knowledge tree). A *Current* view lays out the implemented
  chain in two owner lanes (Daniele's signal→EMD→bands front end handing off to
  Giulio's signed-descriptive → unsigned-filtering → LRG → metrics back end); a
  *Proposed* view shows the full CO₂-vs-rest comparison vision converging on the
  differential-node lens. Pure React/SVG/CSS, no backend (`views/PipelineView.tsx`).
- **Done**: the **Brain** tabs — nilearn interactive 3-D connectome / markers brain
  and the LRG partition glass brain, for both the Schaefer-100 and HarvardOxford-48
  atlases.
- **Done**: the **Compare** tab — CO₂ vs rest, per-patient vs average, across
  variants / bands, two results side by side.
- **Done**: the **drop-a-folder ingestion workflow** — `python quickstart.py`
  detects each `raw_data_<id>` folder and computes + caches its result bundles
  (`scripts/raw_ingest/`); see `../SETUP.md`.
- **Planned**: the full CO₂-vs-rest differential-node lens (the *Proposed*
  Pipeline view). See the project plan for the roadmap.
