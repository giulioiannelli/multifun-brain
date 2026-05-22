# 2026-05-20 — April-batch analysis scaffold (initial)

## What changed

Created the loader, orchestration scripts, and presentation notebooks
needed to start analysing the new April 2026 correlation-matrix batch
that Daniele dropped at `data/correlation_mat_april_data/` on 2026-04-23.

## New code

- `multifunbrain/datasets/april.py` — metadata-aware loader. Public API:
  - `AprilEntry` dataclass: `level`, `contrast`, `processing`, `band`,
    `subject`, `path`.
  - `discover_april(root, levels=None)` — walks the three aggregation
    subdirectories (`freq-contrast-global/`, `freq-contrast-inter/`,
    `freq-user-constrast-inter/`), parses filenames, sorts, excludes
    `.DS_Store`, `All_Emd_diz.pkl`, `hist_C_I/`.
  - `load_entry(entry)` — wraps `multifunbrain.io.corrmatrix.load_correlation_matrix`.
    Single place to add a structural adapter if the pickles turn out to
    be dict-wrapped (not bare arrays); discovered at runtime.
  - `entries_to_dataframe(entries)` — manifest DataFrame.
  - `label_for(entry)` — pipeline-result label string.

- `test/test_datasets_april.py` — data-independent tests for filename
  parsing, label generation, discovery on synthetic trees.

- `scripts/april/` — orchestration scripts (≤80 lines each, argparse-
  based, thin):
  - `00_inventory.py` — discover + write `manifest.csv`.
  - `01_run_global.py` — pipeline on 10 global aggregates → results.pkl + summary.csv + config.json.
  - `02_run_bands.py` — pipeline on 30 band aggregates.
  - `03_run_patients.py` — pipeline on 180 per-subject × band files (with `--subject` filter).

- `notebooks/april/` — presentation-only notebooks (load `results.pkl`
  via `load_results`, plot, discuss):
  - `01_global_overview.ipynb`
  - `02_band_comparison.ipynb`
  - `03_patient_variability.ipynb`
  - `04_contrast_comparison.ipynb`

## Order of analysis (per user direction)

1. **Global aggregates** (10 matrices). Smoke-test entry — confirms
   loader and pipeline on the new data shape before scaling up.
2. **Band aggregates** (30 matrices).
3. **Per-subject × band** (180 matrices).

Failures in any of the three layers are captured in
`failed_matrices.json` (existing pipeline convention) — not silently dropped.

## Gamma calibration

`PipelineConfig.gamma = n_regions / n_timepoints` is needed for MP
overlay + MP-validated filtering. The April matrices already collapse
`n_timepoints`. **Action item**: ask Daniele for `n_timepoints` per
acquisition. Fallback: run with `gamma=None`, document the choice in
`.claude/reports/<date>_april-*.md` once we have results.

## Out of scope (deferred)

- `All_Emd_diz.pkl` (150 MB combined dataset) — handle after per-file
  pickles are validated.
- Permutation tests for CO2-vs-rest statistical contrast — comes after
  the descriptive pass.

## Verification

- `pytest test/ -q` — 62 prior tests + new `test_datasets_april.py` tests pass.
- Smoke: `python scripts/april/00_inventory.py` produces a manifest CSV
  with **220 rows** (10 global + 30 band + 180 patient; `All_Emd_diz.pkl`,
  `.DS_Store`, and `hist_C_I/` PNGs excluded). Confirmed at the verification step.
