# 2026-06-24 — Signal-tab EMD parity with Daniele's notebook + raw-data renaming

Branch: `dashboard`. Triggered by receiving Daniele's reference notebook
(`EMD_Schaefer_April_2026_ordinato.ipynb`) and verifying the dashboard's Signal
tab against it.

## Verification finding

Our Signal-tab EMD re-derivation differed from Daniele's notebook on the three
parameters that decide IMF→band assignment (all three were the choices made
*before* we had his notebook):

| Aspect | Notebook (authoritative) | Was (dashboard) |
|---|---|---|
| Sampling rate | `fs = 1/TR`, **per variant**: `bpf*` 1.353 s, `optcom/MIR*` 0.98 s → **Hz** | `sample_rate=1.0` (cycles/sample) |
| Band edges | **fixed** canonical Hz: s5 `[0.010,0.027]`, s4 `[0.027,0.073]`, s\* `[0.073,0.180]` | data-driven geometric midpoints |
| Char. frequency | `np.nanmedian(inst_freq)` (HHT) | amplitude-weighted mean |

(Band aggregation sum-vs-mean is immaterial: `corrcoef` is per-row
scale-invariant, so the per-band matrices are identical either way. The
Correlation tab's per-band matrices already come from Daniele's own hand-off
(`data/correlation_mat_april_data/`), so only the Signal tab's *live* recompute
diverged.)

User chose: **Daniele's method as default + data-driven as a toggle**.

## What changed (code)

- **`multifunbrain/io/timeseries.py`** — `SAMPLING_INTERVAL_SECONDS`
  (processing→TR) + `sampling_rate(processing) -> fs` (Hz; unknown → 1.0). Both
  re-exported from `multifunbrain.io`.
- **`multifunbrain/analysis/emd_bands.py`** —
  - `CANONICAL_BANDS` (fixed Hz edges) + `canonical_scheme(centers=None)`.
  - `sift_with_frequencies(signal, sample_rate=1.0, method="median")` — median
    instantaneous frequency is now the default (Daniele); `"weighted_mean"` kept.
  - `assign_imfs` switched to **closed** intervals `[lo,hi]`, first-match
    low→high — exactly reproduces the notebook's nested `if 0.010<=f<=0.027 …
    elif 0.027<f<=0.073 …` for the contiguous canonical bands.
- **`dashboard/backend/timeseries.py`** — frequencies computed in Hz via
  `sampling_rate(processing)`; cohort pool (`_cohort_pool_cached`,
  scheme-independent) split from scheme building (`_scheme_for`), so toggling
  canonical↔data-driven is free (no re-sift). `cohort_bands` /
  `band_reconstruction` take `scheme="canonical" | "data_driven"`. Canonical
  reconstruction skips the cohort pool (fixed edges).
- **`dashboard/backend/serializers/signal.py`** — `cohort_bands_spec` emits Hz,
  filters `0<f<1` (notebook), bins `ceil(log2 n + 30)`, adds a **period**
  histogram (`period_counts/period_edges`) + `scheme`/`units`/`sample_rate`.
- **`dashboard/backend/routes/signal.py`** — `scheme` query param on
  `/signal/cohort_bands` and `/signal/{kind}` (signal_bands).
- **Frontend** — `buildCohortBands(spec, {yLog, variant})` renders the frequency
  panel or its 1/f **period** companion (bands stored in Hz, converted for
  period); titles show the scheme. `SignalView` gains a **Bands**
  (Canonical/Data-driven) toggle threaded into cohort + reconstruction requests,
  a second cohort period card, and Hz/scheme captions.
- Tests: `test_emd_bands` (+canonical scheme, +closed-edge assignment matching
  the notebook, +median default/weighted-mean available); `test_io_timeseries`
  (+per-variant `sampling_rate`). Full suite **135 passed**; ruff clean;
  `npm run build` clean.

## Validation (cohort median-IF in Hz vs canonical bands)

- **bpfBOLD** (fs 0.739): IMF1 0.119→S\*, IMF2 0.049→Slow-4, IMF3 0.019→Slow-5.
- **optcom_bold** (fs 1.020): IMF1 0.205→(above S\*), IMF2 0.095→S\*, IMF3
  0.037→Slow-4, IMF4 0.013→Slow-5 — the +1 IMF shift is exactly why the
  per-variant TR matters (the old cycles/sample value would mis-bin).
- **Data-driven vs canonical agree**: data-driven edges for co2/bpfBOLD
  (`s5 0.009–0.030, s4 0.030–0.076, s* 0.076–0.185`) ≈ canonical
  (`0.010/0.027/0.073/0.180`).

## Raw-data renaming (`raw_data_<atlas><batch>`)

Three raw `.ts.1D` sets, renamed for at-a-glance identification (all under
gitignored `data/`):

| Old | New | Atlas / scheme |
|---|---|---|
| `raw_data` | `raw_data_schaefer100_april2026` | Schaefer 100, BIDS — Signal-tab default |
| `atlas_timecourses_Schaefer` | `raw_data_schaefer100_november2025` | Schaefer 100, AFNI `kw…` (+`discarded/`) |
| `atlas_timecourses` | `raw_data_harvardoxford48` | HarvardOxford 48, AFNI `kw…` |

`config.RAW_DATA_ROOT` default repointed to the April set. README §4.4 +
`.claude/guide/data-layout.md` updated.

## Follow-up — Signal-tab dataset selector (same day)

User noticed the Signal tab had no **Dataset** dropdown (unlike Correlation /
Network). Added one, supporting all three raw sets.

- **`multifunbrain/io/timeseries.py`** — reader generalized to two schemes. New
  `kw` regex (`<atlas-prefix>_<modality>.ts.1D`); `parse_timecourse_filename`
  tries BIDS first, then kw (returns `subject=None`, no contrast/run, +`atlas`
  id). `TimecourseFile` fields `session/contrast/run` now optional, `+atlas`.
  `discover_timecourses` fills `subject` from the parent `sub-…` dir for kw
  files, skips a `discarded/` subtree, and **skips 0-byte files** (some batches
  ship empty `*4D` placeholders). Tests: `test_parse_filename_kw_scheme`,
  `test_discover_kw_subject_from_parent_and_skips_discarded` (+ existing BIDS
  test now also asserts `atlas`).
- **`dashboard/backend/timeseries.py`** — dataset-aware. `raw_datasets()`
  discovers `raw_data_*` dirs under `DATA_ROOT` → dropdown list (id, n_subjects,
  has_contrast); `_root_for(dataset)` validates the id against discovered dirs
  (no path traversal). All functions take a `dataset` param; `_index` /
  `_n_regions` keyed per-root. `region_names(dataset)` returns Schaefer names
  when the region count is 100, else `region-N` (HarvardOxford 48). `_safe_load`
  wraps every load so a malformed/empty file degrades to 404, never a 500.
- **`dashboard/backend/routes/signal.py`** — `dataset` query param on all signal
  endpoints; `contrast` now optional (the client drops empty params, and kw sets
  have none).
- **Frontend** — `SignalCatalog` gains `dataset` / `datasets` / `has_contrast`
  (+`RawDataset`); `signalCatalog(dataset?)`. `SignalView` adds the **Dataset**
  dropdown, refetches the catalog on change, and hides Contrast / Mode toggle /
  Bands toggle / band-reconstruction panel when the dataset has no contrast — so
  the kw sets show carpet + single-channel + EMD only.

Verified (TestClient): catalog offers all three; april `has_contrast=true`;
nov-2025 `has_contrast=false`, 100 regions, Schaefer names, 110 files (empty
`ckwBOLD4D` dropped); harvardoxford48 48 regions, `region-N` names; carpet + EMD
return 200 on the kw sets with no contrast param. pytest **137 passed**, ruff
clean, `npm run build` clean. Live dev server (`:8001`, reload mode) picked it up.

## Follow-up — HarvardOxford channel names (same day)

The HarvardOxford-48 set showed generic `region-N` channel labels (the only
atlas `region_names` knew was Schaefer-100). The authoritative FSL label file is
already on disk — `data/HarvardOxford-Cortical.xml` (48 `<label index=…>Name`
entries, indices 0–47, matching the 48 columns) — so this was purely a missing
parser, **backend-only** (the frontend already renders `region_names` /
`spec.names`).

- **`dashboard/backend/config.py`** — `HARVARD_OXFORD_XML` (default
  `DATA_ROOT/HarvardOxford-Cortical.xml`, env `MFB_HARVARD_OXFORD_XML`). Also
  refreshed the stale `RAW_DATA_ROOT` comment (it predated the dataset selector).
- **`dashboard/backend/atlas.py`** — `harvard_oxford_names(xml_file=None)`:
  parses the FSL XML, returns names sorted by `index`. Returns `()` when the file
  is absent/malformed (`data/` is gitignored, so the XML ships with the data, not
  the repo) → callers degrade to `region-N` instead of raising.
- **`dashboard/backend/timeseries.py`** — `_atlas_for(root)` resolves the atlas
  id from the discovered filenames (`schaefer100` / `harvardoxford48`).
  `region_names` now: HarvardOxford → XML names (when present), Schaefer → order
  file, else count-based / generic. Robust to a missing XML.
- Test `test/test_dashboard_atlas.py` (parser: index-order sort, missing file,
  malformed XML — all via a temp XML, no dependency on gitignored data).

Verified live (`:8001`): `harvardoxford48` catalog + carpet `names` =
`Frontal Pole … Occipital Pole` (48, none generic); EMD channel 44 →
`"Heschl's Gyrus (includes H1 and H2)"`; Schaefer sets unchanged. pytest **140
passed**, ruff clean. This supersedes the "`region-N` names" note above.
