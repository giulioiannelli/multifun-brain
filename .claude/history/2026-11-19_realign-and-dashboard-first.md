# 2026-11-19 — Realignment, green CI, dashboard-first conventions

## What changed

- **CI fixed (PR #10).** CI had failed on every run since phase 0.
  `multifunbrain/__init__.py` eagerly imported `multifunbrain.notebook`,
  which imports `emd` and `seaborn` (not core deps), so `import
  multifunbrain` failed in a clean `.[dev]` env. The eager import is gone
  (callers use `from multifunbrain.notebook import *`); CI now installs
  `.[dev,dashboard]` and lints `dashboard/backend/`. Green on 3.10–3.12.
- **Phase 2 realigned (PR #9).** `main` (38 commits ahead, all dashboard
  work) merged into `phase-2-plotting-templates` without conflicts. April
  notebooks and guides now import from the `plotlib` package instead of
  the deleted `pipeline_plots.py` / `lrg_multiscale_plots.py`.
  `docs/development.md` repo tree rewritten (it still listed the
  phase-1-retired `analysis/{lrglib,netmetrics}.py` and `core.py`).
- **Dashboard-first conventions.** New `guide/dashboard.md` (backend
  serializer ↔ frontend builder contract, recipes, gotchas, verification
  in the running app). New rules: *always* present in the dashboard,
  *always* verify a dashboard change in the browser, *never* replace the
  nilearn 3-D brain. `CLAUDE.md` / `best-practices.md` updated: notebooks
  are scratch, `main` is the integration branch.

## Why

The user and collaborators decided (2026-11-19) that from now on results
are consumed only through the dashboard GUI. The agent infrastructure
described the library well but said nothing about the dashboard, which is
now ~9k lines and where most work happens.

## Known leftovers

- `notebooks/april/*` data paths are stale: bundles moved from
  `data/correlation_matrices_results/april/` to `…/schaefer100_april2026/`
  and the `lrg_global/` outputs are gone. Treated as legacy, not fixed.
- Phase 3 (CLI split) and phase 6 (`notebooks/` → `labs/`) not started.
