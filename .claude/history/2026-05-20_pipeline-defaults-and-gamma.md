# 2026-05-20 — Pipeline-default change + per-variant gamma

## What changed

### `PipelineConfig` defaults (`multifunbrain/pipeline/config.py`)

| Field | Before | After |
|---|---|---|
| `filter_methods` | `["absolute", "positive", "negative"]` | `["absolute", "partial_correlation"]` |
| `run_community_detection` | `True` (Louvain on) | `False` (Louvain off) |
| `filter_threshold` | `None` (percolation auto) | unchanged |

Rationale: per the user's standing feedback (saved as
`feedback_pipeline-defaults.md` in auto-memory), Louvain on these
signed-correlation networks is not informative; LRG hierarchical
partitions are the meaningful clustering output. Dense / unfiltered
matrices produce graphs LRG cannot meaningfully analyse — the
percolation-based absolute-threshold backbone preserves connectivity
guarantees while extracting structure. The precision-matrix path
(`partial_correlation`) runs in parallel as a complementary
noise-robust route (cf. arxiv 2302.02951).

### Per-variant `gamma` lookup in `multifunbrain/datasets/april.py`

The April batch's raw `.ts.1D` files (located in
`data/correlation_mat_april_data/20260326_raw-data_Maria/`, atlas
Schaefer 2018 100/17Networks → `p = 100`) showed two distinct
`n_timepoints` values:

| Processing variant | `n_timepoints` | `gamma = p/n` |
|---|---|---|
| `bpfBOLD`, `bpfVASO` | 442 | ≈ 0.226 |
| `optcom_bold`, `optcomMIRDenoised_bold`, `MIRNoise_bold` | 597 | ≈ 0.168 |

New public API:

- `N_PARCELS: int = 100`
- `N_TIMEPOINTS_PER_PROCESSING: dict[str, int]`
- `gamma_for(entry: AprilEntry) -> float | None`

### Orchestration update (`scripts/april/_common.py`)

`run_entries` now resolves gamma **per entry** via `gamma_for(entry)`
and builds a per-entry `PipelineConfig`. CLI `--gamma <float>`
overrides globally. Calls `run_pipeline` per matrix instead of
`run_pipeline_batch` (the batch helper takes a single shared config).

Also added a `--run-community-detection` flag so a Louvain comparison
remains one CLI flag away.

## Why

User feedback during the same session, after seeing the initial scaffold:

> *"1. dense matrices will produce not nice graphs to be analyzed with
> the lrg, maybe we should use the percolation threshold analysis ...
> 2. we might like to proceed parallely with the precision matrix ...
> 3. avoid louvain, completely pointless analysis."*

And, separately, pointing to the Maria raw-data folder so gamma could
be computed exactly.

## Caveat surfaced from the raw-data readme

Maria's labels assign `task-co2` to the first run and `task-rest` to
the second run per session. This is a run-index proxy, not a verified
contrast. Flagged in `.claude/reports/` template — any CO2-vs-rest
finding must be cross-checked against the acquisition protocol before
publication.

## Verification

- `pytest test/` → **78 passed** (62 baseline + 12 prior April tests +
  4 new gamma tests).
- `ruff check` on touched files → clean.
- `scripts/april/00_inventory.py` still produces a 220-row manifest.
- Smoke imports: `from multifunbrain.datasets.april import gamma_for, N_PARCELS, N_TIMEPOINTS_PER_PROCESSING` works.

## What's still open

- Apply the same per-variant gamma approach to the older
  `data/correlation_matrices/` batch when we need RMT on it.
- Decide whether to add a fourth filter method (`mp_validated`) by
  default once gamma is reliably set — currently still opt-in via
  `filter_methods=[..., "mp_validated"]`.
- Write the actual analysis report under `.claude/reports/` once the
  pipeline has run on the April global slice.
