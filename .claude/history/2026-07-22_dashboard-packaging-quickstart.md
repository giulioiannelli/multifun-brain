# 2026-07-22 — One-command cross-platform dashboard packaging

Branch: `dashboard`. Goal: collaborators go from a fresh `git clone` to the
dashboard in their browser in the fewest possible steps, identically on
Windows / macOS / Linux, with the setup **auto-detecting the shared data folder**
and caching whatever results aren't already computed.

## New: `quickstart.py` (repo root)

A single **pure-standard-library** entry point (must run *before* deps exist, so
no third-party imports at module load — the library is imported lazily inside
`inventory()`). One command does the whole chain:

1. Python-version check → `pip install -e .[dashboard]` into `sys.executable`
   (skipped when `multifunbrain`/`fastapi`/`uvicorn`/`emd` already import).
2. **`inventory(data_root, results_root)`** — pure, unit-tested: scans
   `data/raw_data_<id>` folders, detects each set's atlas (`atlas_of`), subject
   count, processing-variant count (via `discover_timecourses`), and whether its
   `global`/`bands`/`patients` bundles exist. Prints a table.
3. **`ensure_bundles`** — ingests only datasets whose bundles are missing
   (`python -m scripts.raw_ingest.run_dataset --dataset <id>`), with a clear
   time warning. Skips entirely when everything is cached. Idempotent.
4. **`ensure_frontend`** — `npm install` + `vite build` only when `dist/` is
   absent or `--rebuild`. Degrades gracefully when Node is missing but a build
   exists; clear guidance (use the conda env) when neither.
5. **`serve`** — port-check (refuses, never kills — mirrors `run.sh`), launches
   `uvicorn` in a subprocess, opens the browser.

Modes/flags: `serve` (everyday launch after `git pull`), `--dry-run` (prints the
plan, changes nothing), `--data PATH` (`data_env()` repoints every `MFB_*` root at
a folder living outside the repo — nothing copied), `--no-ingest`, `--rebuild`,
`--no-serve`, `--no-open`, `--reload`, `--port`, `--host`.

Cross-platform care: `shutil.which("npm")` (+ Windows `.cmd` shell fallback),
`webbrowser.open`, socket port-probe, `sys.executable` for every subprocess (so
pip/ingest/serve all hit the *same* interpreter). `importlib.invalidate_caches()`
after a fresh install so the just-installed editable package imports in-process.

## New: `SETUP.md` (repo root)

The complete clone→browser guide: conda one-step env (Python **and** Node, all
OSes), where the gitignored `data/` folder goes (or `--data`), first-run compute
time, the `git pull` + `serve` update loop, LAN sharing (`--host 0.0.0.0`), and a
troubleshooting table (busy port, missing Node, Windows `run.sh`-is-bash-only).
`README.md` gets a dashboard callout; `dashboard/README.md` now leads with the
quickstart and frames itself as the deep reference.

## Rationale

- **Node is the only cross-platform friction** → solved by the existing conda env
  bundling `nodejs`, not by committing `dist/`.
- Ship-raw vs ship-bundles both work: quickstart detects computed bundles and
  skips ingest (fast "look at results"), or computes+caches from raw once
  (the "start from the schaefer folder, automatic" path). Same command either way.

## Verify

`test/test_quickstart_inventory.py` (6 tests: atlas detection, bundle-status /
partial-bundle / empty inventory, `data_env` existing-target filtering + raw-dir
fallback). Full suite **192 pass**, ruff clean. Live: `--dry-run` prints the real
3-dataset inventory; a real `serve` on a throwaway port returned `200` for the UI
and `/api/catalog`. Committed to branch `dashboard`; the project's GitHub URL was
also set to `giulioiannelli/multifun-brain` (was the `your-org` placeholder).
