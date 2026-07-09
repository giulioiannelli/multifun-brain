# Data layout

`data/` is gitignored. Files come from collaborators. Treat all pickles
as **untrusted** — never unpickle in a sandbox check (use the
metadata-aware loaders in `multifunbrain.datasets.*` to read them inside
the analysis pipeline, where the pickle deserialisation is the
intentional action).

## Top-level structure

```
data/
├── raw_data_schaefer100_april2026/          # raw .ts.1D time-series — Schaefer 100, BIDS scheme (Signal-tab default)
├── raw_data_schaefer100_november2025/       # raw .ts.1D — Schaefer 100, Nov 2025, AFNI `kw…` scheme (was atlas_timecourses_Schaefer)
├── raw_data_harvardoxford48/                # raw .ts.1D — HarvardOxford 48, AFNI `kw…` scheme (was atlas_timecourses)
├── correlation_matrices/                    # older / baseline batch (per-contrast variants)
├── correlation_matrices_old/                # archived earlier version
├── correlation_matrices_results/            # pipeline outputs
│   ├── per_matrix/                          # per-matrix diagnostic outputs
│   ├── results.pkl                          # historical aggregated results
│   ├── summary.csv
│   ├── failed_matrices.json
│   └── april/                               # ← outputs for the April batch (see scripts/april/)
├── correlation_mat_april_data/              # ← April 2026 batch (Daniele)  ← active dataset
├── figures/                                 # rendered figures
├── fsl/                                     # atlas + masks
├── neuroplots/                              # plot assets (atlases for visualisation)
├── schaefer_2018/                           # Schaefer parcellation
├── HarvardOxford_48Parcels.ts.1D
├── HarvardOxford-Cortical.xml
└── kwfurN_Bold_IMF_frequencies.pkl
```

## April batch (`data/correlation_mat_april_data/`)

Dropped by Daniele on **2026-04-23**, total **169 MB**, **226 pickle
files** + 10 histogram PNGs + 1 mega-aggregate (`All_Emd_diz.pkl`,
150 MB, deferred).

### Stratification

- **Contrasts** (2): `co2` (hypercapnia), `rest` (resting state).
- **Processing variants** (5):
  - `bpfBOLD` — bandpass filtered BOLD
  - `bpfVASO` — bandpass filtered VASO
  - `MIRNoise_bold` — BOLD with MIR noise handling
  - `optcom_bold` — optimal combination BOLD
  - `optcomMIRDenoised_bold` — optimal combination, MIR-denoised
- **Bands** (3, IMF slow modes): `s4`, `s5`, `sstar`.
- **Subjects** (6): `sub-00246757`, `sub-00259685`, `sub-00307729`,
  `sub-00308305`, `sub-VA11266`, `sub-VA9757`.

### Aggregation levels (subdirectories)

| Path | Files | What |
|---|---|---|
| `freq-contrast-global/<contrast>_<proc>/<contrast>_<proc>_GLOBAL.pkl` | 10 | One matrix per contrast×processing, aggregated across all subjects and bands. Smoke-test entry point. |
| `freq-contrast-inter/<contrast>_<proc>/<contrast>_<proc>_<band>.pkl` | 30 | Per-band aggregates across subjects (2 × 5 × 3). |
| `freq-user-constrast-inter/<subject>/<contrast>_<proc>/<subject>_<contrast>_<proc>_<band>.pkl` | 180 | Per-subject × contrast × band (6 × 10 × 3). |
| `hist_C_I/*.png` | 10 | Histogram visualisations, one per (contrast, processing). |
| `All_Emd_diz.pkl` | 1 | 150 MB consolidated EMD dataset — handle after per-file pickles are validated. |

Use `multifunbrain.datasets.april.discover_april()` to walk this tree
and produce a list of metadata-tagged `AprilEntry` records. The
orchestration scripts under `scripts/april/` consume those entries and
feed them into the pipeline.

## Older batches (`data/correlation_matrices/`)

Contains per-contrast variants (`freq-c`, `freq-c_glm`, `freq-c_newp`,
`freq-users`, `freq-users_glm`, `freq-users_newp`, ...). Kept for
comparison studies; not currently the active analysis target.

## Raw time-series — Maria's folder

`data/correlation_mat_april_data/20260326_raw-data_Maria/` holds the
raw AFNI `.ts.1D` files that produced the correlation pickles. From
its `readme_by_chatGPT.txt`:

- **Atlas**: Schaefer 2018, 100 parcels, 17 networks → `p = 100`.
- **Time points**:
  - `bpfBOLD`, `bpfVASO` runs: `n = 442` → `gamma ≈ 0.226`
  - `optcom_bold`, `optcomMIRDenoised_bold`, `MIRNoise_bold` runs:
    `n = 597` → `gamma ≈ 0.168`
- **Caveat on contrast labels**: Maria labels runs by *order* (first
  run = `co2`, second = `rest`). Treat the contrast label as a
  run-index proxy until cross-checked against the acquisition sheet.

The values live in
`multifunbrain.datasets.april.N_TIMEPOINTS_PER_PROCESSING` and are
applied per entry by `gamma_for(entry)`. Add new processing variants
there if more data arrives.

## `gamma` calibration

`PipelineConfig.gamma = n_regions / n_timepoints` is required for the
Marchenko–Pastur edge and the MP-validated filter. For the April
batch, `gamma_for(entry)` (see above) supplies the right value per
processing variant. The orchestration scripts under `scripts/april/`
use it automatically; pass `--gamma <float>` to override globally.
