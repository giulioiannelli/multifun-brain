# 2026-05-20 — Package reorganization

## What changed

Split the flat `multifunbrain/analysis/*.py` modules + the monolithic
`pipeline.py` into scope-based subpackages. Each function/class now has
exactly one canonical home; old import paths continue to work through
thin re-export shims.

## New top-level subpackages

- `multifunbrain/io/` — file I/O (`corrmatrix.py`, `results.py`).
- `multifunbrain/preprocessing/` — matrix cleaning, dead-region
  detection, MP denoising (`dead_regions.py`, `prepare.py`,
  `denoising.py`).
- `multifunbrain/processing/` — signed→unsigned filtering, backbone,
  partial correlation, percolation, temporal filtering
  (`filtering.py`, `backbone.py`, `partial_correlation.py`,
  `percolation.py`, `temporal.py`, internal `_giant.py`).
- `multifunbrain/datasets/` — dataset-specific loaders (April batch
  arrives next).
- `multifunbrain/pipeline/` — orchestrators split into `config.py`,
  `result.py`, `runner.py`, `discovery.py`.

## Split within `analysis/`

- `analysis/descriptive/` (package) — `weights.py`, `spectrum.py`,
  `signed.py`, `report.py`. Replaces the old flat
  `analysis/descriptive.py`.
- `analysis/lrg/` — `kernel.py`, `distance.py`, `partitions.py`.
  `lrglib.py` is retained as a re-export shim.
- `analysis/network/` — `global_metrics.py`, `node_metrics.py`,
  `community.py`, `distribution.py`, `report.py`. Replaces
  flat-module functions; `netmetrics.py` is now a shim.
- `analysis/partition.py` — `adjusted_rand_index`, `compare_partition_sets`
  (extracted from the legacy `corrmatrix.py`).

## Back-compat shims (still importable from old paths)

- `multifunbrain/analysis/corrmatrix.py` — re-exports from `io.corrmatrix`,
  `preprocessing.{dead_regions,prepare,denoising}`, `analysis.lrg.partitions`,
  `analysis.partition`.
- `multifunbrain/analysis/filtering.py` — re-exports from
  `processing.{filtering,backbone,percolation}`.
- `multifunbrain/analysis/netmetrics.py` — re-exports from `analysis.network`.
- `multifunbrain/analysis/lrglib.py` — re-exports from `analysis.lrg`.
- `multifunbrain/core.py` — re-exports `band_filter` from
  `processing.temporal`.
- `multifunbrain/pipeline.py` was deleted (path is now a package);
  `multifunbrain/pipeline/__init__.py` re-exports the same public names.

## Why

The user's directive was: *"each function and class in its own path and
well organized and reusable so that we avoid to reinvent the wheel all
the times. ... void any duplicate of submodules and stuff each thing
has to be organized and unique."* The flat layout was making it hard
for new agents to find canonical locations, and several files
(`corrmatrix.py`, `descriptive.py`, `pipeline.py`) had clearly mixed
concerns.

## Verification

- `pytest test/ -q` → **62 passed** before, during (after each step),
  and after the reorg. Zero test modifications were needed — the
  shim strategy worked end-to-end.
- `ruff check` on the new submodules: **0 errors** after a 3-issue
  auto-fix pass. Pre-existing 82-error baseline in legacy modules
  remains (out of scope today).
- Smoke imports verified both canonical and legacy paths:
  `from multifunbrain.io.corrmatrix import load_correlation_matrix`
  and `from multifunbrain.analysis.corrmatrix import load_correlation_matrix`
  both work.

## What's still open

- The legacy shim files in `analysis/` will be removed eventually
  (separate cleanup PR) once we've migrated downstream notebooks to
  canonical paths.
- `pyproject.toml` ruff config still uses deprecated top-level keys.
- Pre-existing 82 ruff lints in legacy modules — not addressed today.
- `docs/development.md` has a stale "Repository layout" block; should
  be refreshed (see `.claude/guide/module-map.md` for the current map
  in the meantime).

## Files touched

See the diff for the `april-scaffold` branch against `main`.
