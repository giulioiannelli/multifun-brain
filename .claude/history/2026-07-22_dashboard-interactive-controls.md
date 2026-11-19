# 2026-07-22 — Dashboard interactive controls + dataset unification

Branch: `dashboard`. A large pass over almost every result tab, making the plots
interactive/tunable, plus finishing the deferred **dataset-unify** work so the
result tabs offer the same three raw datasets as the Signal tab.

## Interactive controls (per tab)

- **Signal — average signal.** New "⟨mean over regions⟩" entry in the channel
  picker (sentinel channel `-1`). Flows through the single-channel, EMD and
  band-reconstruction panels. Backend: `serializers/signal.py` `_channel_name` +
  `channel_spec` handle `-1`; `timeseries._sift_cached` sifts `ts.mean(axis=0)`.

- **Correlation — exclude noisy channels.** New `ChannelExcluder` dropdown
  (searchable checkboxes with atlas-colour swatches). Excluded indices are sent
  as `?exclude=i,j,...`; `serializers/descriptive._working_matrices` drops those
  rows/cols from `corr_prepared` (or the MP-cleaned matrix) and **recomputes**
  heatmap / precision (inverse of the unit-diag submatrix) / spectrum (eig of the
  submatrix, `gamma` rescaled to the retained channel count) / weights. New
  lightweight `region_names` serializer feeds the selector's label list.

- **Network — layouts + colour/size + sparsification.** `NetworkGraph` rewritten:
  - Layouts: server presets `spring`/`kamada`/`spectral` (shipped in
    `spec.layouts`, aligned to node order) + Cytoscape client layouts
    `cose`/`concentric`/`circle`/`grid`/`breadthfirst`. Preset changes remount;
    colour/size/scale restyle in place via a `cy` ref + `useEffect` (no relayout).
  - Colour by atlas-network / degree / strength (blue ramp); size by
    strength / degree / uniform + Node× scale; edge width via Edge× scale.
  - **Sparsify** = `percolation` / `disparity` / `|r|`-threshold re-derived from
    `corr_prepared` with the pipeline's own `filter_validated` /
    `filter_absolute_threshold` (`network._derive_graph`), + an Edges display
    quantile. Note: the "absolute" filtered network is ~0.90 density here, so
    percolation barely thins it (θ\* keeps ~90%) and disparity collapses — the
    `|r|`-threshold + edge quantile + a spread layout are the useful de-hairballers.
    Metrics panels still reflect the pipeline's stored filter (noted in the caption).

- **LRG — τ/cut sliders, coloured dendrogram, legible flow/Sankey.** New
  `views/LrgView.tsx` owns two sliders: **τ step** over the precomputed grid
  (already spans `[1/λmax, τmax]`; shown in a note) and a geometric **cut h**
  slider. `dendrogram_spec` takes `cut_height` (defaults to the LRG
  `flat_threshold`), colours branches per-cluster below the cut via
  `set_link_color_palette` + `color_threshold`, and reports `n_clusters_at_cut`
  (`fcluster`). The partition-flow and Sankey are no longer a confusing replica:
  `_canonical_labels` relabels clusters for **cross-τ colour continuity** (greedy
  max-overlap, splits kept distinct); the flow raster sorts rows into merging
  bands, and the Sankey lays nodes in τ columns (`node_x`) coloured by community
  with source-coloured links + τ annotations.

- **Brain 3-D — larger nodes.** `node_size` param through
  `routes/brain3d.py` → `brain3d.render/_build_html` (markers ×1.3), plus an
  S/M/L/XL control. Default bumped 3→9.

## Dataset unification

- **Renamed** `data/correlation_matrices_results/april` →
  `schaefer100_april2026` (gitignored data; id now matches the raw set).
  `App.tsx` preferred default updated.
- **`catalog.parse_label`** contrast-safe: the contrast prefix is only peeled off
  for `co2`/`rest`, so kw variant tokens (`kwoptcomMIRDenoised_bold`,
  `clean_kwCBF4D`, …) keep their full name with `contrast=None`. Added the 3-part
  `patient/<subj>/<variant>` (no-band) form.
- **`catalog.list_datasets`** now filters bundles to those whose top-level folder
  is a known raw dataset id (`_known_dataset_ids` ← `timeseries._raw_data_dirs`),
  so the result-tab dataset list mirrors the Signal tab and excludes legacy junk.
  Skipped when no raw datasets are present (results-only checkout).
- **`dashboard/backend/timeseries.py`** Signal-tab EMD (`sift_channel`,
  `_cohort_pool_cached`, `band_reconstruction`) now resolves fs via
  `sampling_rate_for(root, n_timepoints, processing)` so the kw sets report Hz
  from their `acquisition.json` TRs.

## Raw ingest (compute for the kw sets)

- **New `scripts/raw_ingest/{__init__,_common,run_dataset}.py`.** For a
  `raw_data_<id>` tree: per `(subject, variant)` → `load_timecourses` →
  `band_correlation_matrices(ts, sampling_rate_for(...))` → `run_pipeline` on
  each of `{full,s5,s4,sstar}` (repo-default `PipelineConfig`: percolation filter,
  LRG on, no Louvain), wrapped so a degenerate band records `result.error` instead
  of crashing. Per-variant group means → `global`/`bands` bundles; per-subject →
  `patients` bundle. Labels mirror April but contrast-less: `global/<variant>`,
  `band/<variant>/<band>`, `patient/<subj>/<variant>[/<band>]`. Self-contained
  `dump_bundle` writes `results.pkl`/`summary.csv`/`config.json`/`failed_matrices.json`.
- Ran full compute for `schaefer100_november2025` (11 subj × 10 variants) and
  `harvardoxford48` (13 × 4). ASL (N=110, Nyquist 0.091 Hz) sstar bands are above
  Nyquist → recorded as graceful failures.

## Verification

- `pytest test/` green (incl. new `band_correlation_matrices`,
  `load_acquisition_metadata`/`sampling_rate_for`, and `test_dashboard_catalog.py`
  parse_label/list_datasets tests). `ruff check` clean. `tsc --noEmit` +
  `npm run build` clean.
- Live-server curl of every new endpoint (network+sparsify, dendrogram+cut,
  heatmap/spectrum+exclude, region_names, signal channel `-1`, brain3d
  node_size) → 200 with correct payloads.
- Headless-Chrome: Network controls + coloured/sized graph; LRG τ/cut sliders
  recolour the dendrogram; Correlation excluder rows render correctly.

## Notes / follow-ups

- **HarvardOxford region names** in the result tabs are wrong: `remap`
  defaults to the Schaefer atlas, so a 48-node hox result shows Schaefer labels
  and its 3-D brain is unavailable (centroid count mismatch). The
  matrices/analysis are correct; making `remap` atlas-aware per dataset is the
  fix. `schaefer100_november2025` (100 = Schaefer) is unaffected.
- Fresh kw LRG grids have ~30 τ (the runner default when `tau_values=None`);
  April was computed with 6. The τ slider adapts to either.

## Update — 3-axis facets · C(τ)/Ψ(n) · continuous client-side cut

Follow-up pass (same branch/day) driven by three requests: split the kw sets by
contrast, add the LRG diagnostic curves, and make the height cut continuous.

### Three facet axes (task · contrast · processing)

The imaging **contrast** (modality) lives *inside* the variant token, not as a
co2/rest prefix. New `dashboard/backend/facets.py` centralises the parse: a token
→ `(contrast, processing)` where contrast ∈ {bold, vaso, cbf, noise} (substring,
`noise` before `bold`; `denoised` is masked so `optcomMIRDenoised` reads `bold`)
and processing is normalised (clean / optcom / optcomMIRdenoised / bpf / raw /
furN / fcurN / MNI152 / MIRnoise), distinct within each (task, contrast) group.
The co2/rest axis is renamed **task**.

- `catalog.parse_label` now returns `task`+`contrast`+`processing` via
  `facets.parse_variant`; `_FACET_KEYS` gains `task`.
- `SelectorBar.tsx`: adds a Task dropdown (hidden when a set has none, i.e. all
  kw), modality display labels (BOLD/VASO/CBF), new PROC_ORDER.
- Signal tab (`timeseries.signal_catalog` + `SignalView.tsx`): entries carry the
  parsed facets **plus the original `token`**; the frontend resolves a facet
  selection back to its token so file loading is unchanged. `raw_datasets`
  `has_contrast` now reflects modality → the Dataset dropdown's "· no contrast"
  tag is gone. Cohort/band mode stays gated on `has_task` (April only).
- Tests: `test_dashboard_facets.py` (parse + within-group uniqueness),
  updated `test_dashboard_catalog.py`.

### LRG C(τ) + Ψ(n) + client-side cut (no re-ingest)

Both curves and the recolour are computed **online from already-cached data** —
the τ-sweep linkage stays the only cached-and-expensive thing.

- `multifunbrain/analysis/lrg/partitions.py`: extracted
  `prepared_graph_laplacian_spectrum` (the prepare→graph→Laplacian chain, DRY with
  `hierarchical_partitions_from_corr`).
- `serializers/lrg.py`: **`specific_heat_spec`** recomputes the Laplacian spectrum
  from the stored `filtered_networks[f]["graph"]` (the exact LRG Laplacian) →
  `entropy()` → C(τ), τ′, τ*; **`psi_spec`** runs `compute_optimal_threshold` on
  the cached linkage → Ψ(n), optimal cluster count + height. **`dendrogram_spec`**
  rewritten to ship the linkage `Z`, per-drawn-link node id (`link_ids`, via a
  `link_color_func` tag) and `leaves` — and NO server-side `color_list`/`cut_height`.
- `figures.ts`: `colorDendrogram(spec, h)` union-finds merges below `h` client-side
  → per-link colours + cluster count; `buildDendrogram(spec, cutHeight?)`,
  `buildSpecificHeat`, `buildPsi`. `LrgView.tsx`: dendrogram fetch drops
  `cut_height` (so dragging h never refetches — verified: 0 `/api/plot` requests on
  drag), C(τ) fetched once (τ-independent; current-τ marker moves client-side), Ψ
  refetched per τ; a "Ψ-optimal" button snaps the cut to the stable height.
- Test: `test_lrg_dashboard.py` (specific_heat / psi / dendrogram client contract).

### Cross-subject average (Signal tab)

New `AVG_SUBJECT = "__avg__"` sentinel in the Subject dropdown ("Average (all
subjects)"): `timeseries._avg_array_cached` element-wise-means every subject's raw
timecourse for a variant (truncated to the shortest run), and `_full_sift`
(shared by `sift_channel`/`band_reconstruction`) sifts that mean — so carpet,
single channel, EMD and band views all show the average. Backend resolution is
unchanged (per-subject still keys on the file path); `get_timecourses` branches on
the sentinel. Frontend resolves the facet selection to a token subject-independently,
and the fallback effect treats the average as valid whenever the variant exists.
Test: `test_dashboard_signal.py`.

### Note

The LRG feature was first delegated to a `fork` subagent that burned ~268k tokens
and returned a hallucinated orchestrator-narration having implemented nothing —
implemented directly instead. Lesson: verify a fork's *diff*, never its prose.

### Verify

186 pytest pass · ruff clean · tsc clean · npm build clean. Browser: April shows
Task/Contrast/Processing; kw (nov2025/hox) hide Task and show modality Contrast +
no "· no contrast"; C(τ)/Ψ(n) render; the h-slider recolours with zero network;
the Signal-tab "Average (all subjects)" renders the mean carpet/channel/EMD.
