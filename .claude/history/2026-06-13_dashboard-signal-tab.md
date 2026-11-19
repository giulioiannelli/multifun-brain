# 2026-06-13 — Dashboard: Signal tab + raw-timecourse reader

Branch: `dashboard`. Adds a **Signal** tab (raw ROI timecourses, before any
correlation) as the first tab, and renames the former **Descriptive** tab to
**Correlation**. Introduces the canonical library reader for raw `.ts.1D`
timecourses (the user asked for reader helpers *not* buried in `april.py`).

## What changed

### Library — canonical raw-timecourse I/O

- **New `multifunbrain/io/timeseries.py`** — sibling of `io/corrmatrix.py`:
  - `load_timecourses(path, *, region_major=True)` — read an AFNI `3dROIstats`
    `.ts.1D` file (on disk: `n_timepoints` rows × `n_regions` cols). Returns
    `(n_regions, n_timepoints)` by default, matching
    `analysis.corrnet.compute_correlation_matrix`'s convention.
  - `parse_timecourse_filename(name)` — BIDS-like AFNI name →
    `(subject, session, contrast, run, processing)`. Longest-first proc
    alternation so `optcomMIRDenoised_bold` isn't shadowed by `optcom_bold`.
  - `discover_timecourses(root, *, contrasts=, processings=)` →
    sorted `list[TimecourseFile]`.
  - Exported from `multifunbrain.io`. Tests in `test/test_io_timeseries.py`
    (7 cases: parse/disambiguate/orientation/discovery). The old ad-hoc reader
    `scripts/april/_handoff_compare.py::load_subject_ts` stays as-is (script
    local); `datasets/april.py` is untouched — it loads *correlation pickles*,
    not raw TS.

### Dashboard backend — Signal source

- `config.py`: `RAW_DATA_ROOT` (env `MFB_RAW_DATA_ROOT`, default
  `data/raw_data/`).
- **New `dashboard/backend/timeseries.py`** — thin adapter over the library
  reader: `signal_catalog()` (subjects/contrasts/processings + atlas region
  names), `get_timecourses(...)` (LRU by path+mtime), `sift_channel(...)`
  (LRU EMD via `emd.sift.sift`, expensive step cached). Query params resolve to
  a *discovered* file — no path-traversal surface.
- **New `dashboard/backend/serializers/signal.py`** — pure
  `(array|sift, names) → JSON` for `signal_heatmap` (full N×T carpet),
  `signal_channel` (one region), `signal_emd` (signal + IMFs + residual).
- **New `dashboard/backend/routes/signal.py`** —
  `GET /api/signal/catalog` and `GET /api/signal/{signal_heatmap|signal_channel|signal_emd}`;
  wired into `app.py`.
- `pyproject.toml`: `emd>=0.6` added to the `dashboard` extra.

### Dashboard frontend — Signal tab + tab lift

- Tabs lifted from `ExploreView` to **`App`**: `Signal | Correlation | Network |
  LRG` (Signal default). The result `SelectorBar` shows only for the non-Signal
  tabs; Signal has its own selectors.
- **New `views/SignalView.tsx`** — Subject / Contrast / Processing / Channel
  dropdowns + carpet z-score/Raw toggle. Three wide panels: carpet heatmap,
  single-channel timecourse, stacked EMD/IMF waterfall.
- `ExploreView` now takes a `tab` prop (`Correlation|Network|LRG`); the
  former `Descriptive` section is **Correlation**.
- `figures.ts`: `buildSignalHeatmap` (Greys, per-row z-score with raw kept for
  hover), `buildSignalChannel` (line), `buildSignalEMD` (manual stacked y-axis
  domains, shared time axis).
- `usePlot`/`PlotPanel` gained an `endpoint: "plot" | "signal"` option;
  `api.signal` + `api.signalCatalog` + `PLOT_FETCHERS` map in `client.ts`.

## Verification

- `pytest test/` → 126 passed (7 new). `ruff check multifunbrain/ dashboard/backend/`
  clean. `npm run build` clean.
- Headless-Chrome e2e: tabs render `Signal|Correlation|Network|LRG`, Signal is
  default with the 3 cards; carpet=heatmap trace, channel=scatter, EMD=signal+6
  IMFs; switching Channel updates channel+EMD titles live; carpet Raw toggle
  works; Correlation tab shows the SelectorBar + renamed cards. No JS errors.

## Notes / data

- `data/raw_data/`: 6 subjects × 2 contrasts × 5 processing variants = 60
  `.ts.1D` files (gitignored). Raw signal has **no band** — bands are an
  EMD/IMF-derived stage downstream, not present in the raw hand-off.

## Follow-up (same day) — EMD frequency bands

Defines the **s5 / s4 / s\*** bands by reconstructing Daniele's
`hist_C_I/*.png` IMF-frequency histograms, and adds a two-mode Signal tab.

Decisions (asked the user): **recompute from raw** (not from the
RCE-guarded `All_Emd_diz.pkl`), **cycles/sample** axis (no TR supplied), and
**data-driven band edges** (not hard-coded canonical Hz).

- **New `multifunbrain/analysis/emd_bands.py`** (lazy `emd` import):
  `sift_with_frequencies` (sift + amplitude-weighted mean instantaneous freq per
  IMF), `estimate_band_edges` (per-IMF-index cluster medians → geometric-midpoint
  edges; top-3 clusters become sstar/s4/s5, the rest is drift), `assign_imfs`,
  `reconstruct_bands`. `BAND_ORDER=("s5","s4","sstar")`. Tests
  `test/test_emd_bands.py` (5). Validated against Daniele: the cohort clusters
  (cyc/sample) ÷ his Hz peaks ≈ 1.2 s consistently ⇒ implied TR≈1.2 s, IMF1→s\*,
  IMF2→s4, IMF3→s5.
- Backend `timeseries.py`: `_sift_cached` now stores full IMFs + freqs (shared by
  EMD panel and reconstruction); `cohort_bands(contrast,processing)` (LRU, ~3 s
  first call, pools 6×100 ROIs) and `band_reconstruction(...)`. Serializers
  `cohort_bands_spec` / `band_reconstruction_spec`; routes
  `GET /api/signal/cohort_bands` and `signal_bands` kind.
- Frontend: Signal tab gains a **Mode** toggle — *Per subject* (carpet + channel
  + EMD + **band reconstruction** panel) vs *Cohort (EMD bands)* (contrast +
  processing → IMF-frequency histogram with shaded data-driven bands, IMF cluster
  lines, y log/linear). `figures.ts` `buildCohortBands` + `buildBandReconstruction`.
- Verified: pytest 131 pass, ruff clean, npm build clean; headless e2e of both
  modes (cohort histogram = 64 bars + 3 band rects + IMF lines + Slow-5/Slow-4/S\*
  labels; band-recon = signal + 3 stacked band rows). No JS errors.
