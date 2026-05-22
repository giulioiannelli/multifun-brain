# 2026-05-22 — Phase 1: retire analysis shims and collapse core.py

Second phase of the multi-phase cleanup described in
`/home/opisthofulax/.claude/plans/prologue-the-scope-hashed-honey.md`.
Branch: `phase-1-retire-shims` (off `phase-0-hygiene`).

## What changed

### Shim deletions (5 files)

- `multifunbrain/analysis/corrmatrix.py` — deleted; was re-exporting from
  `io.corrmatrix`, `preprocessing.{dead_regions,denoising,prepare}`,
  `analysis.lrg.partitions`, `analysis.partition`.
- `multifunbrain/analysis/filtering.py` — deleted; was re-exporting from
  `processing.{filtering,backbone,percolation}`.
- `multifunbrain/analysis/netmetrics.py` — deleted; was re-exporting from
  `analysis.network.*`.
- `multifunbrain/analysis/lrglib.py` — deleted; was re-exporting from
  `analysis.lrg.*`.
- `multifunbrain/core.py` — deleted; collapsed (see below).

### `multifunbrain/core.py` collapse

- `marchenko_pastur_density` (the canonical implementation) moved from
  `core.py` into `multifunbrain/preprocessing/denoising.py` — its
  natural home, where it was already being re-exported.
- `hello_brain` inlined into `multifunbrain/cli.py::_cmd_hello` (the
  only caller — it's a one-line greeting).
- `band_filter` no longer re-exported from `core.py`; callers now import
  from `multifunbrain.processing.temporal` (or the top-level
  `multifunbrain` namespace).

### Consumer migrations

- `multifunbrain/__init__.py`: replaced `from .core import band_filter,
  hello_brain, marchenko_pastur_density` with imports from canonical
  homes (`preprocessing.denoising`, `processing.temporal`). Dropped
  `hello_brain` from `__all__` (CLI-only now).
- `multifunbrain/analysis/__init__.py`: rewritten to import every
  symbol from its canonical home, no longer through the shim files.
  Public `__all__` preserved so `from multifunbrain.analysis import X`
  keeps working for the curated ergonomic surface (used by the
  `multifunbrain.notebook` wildcard convenience namespace).
- `multifunbrain/analysis/corrnet.py`: `marchenko_pastur_density` alias
  now imports from `..preprocessing.denoising`.
- `multifunbrain/analysis/descriptive/spectrum.py`: same.
- `multifunbrain/generation/generators.py`: `band_filter` import now
  from `..processing.temporal`.
- `multifunbrain/cli.py`: dropped `from .core import hello_brain`,
  inlined the greeting in `_cmd_hello`.

### External callers migrated (tests + notebooks)

The audit found callers the original Explore pass missed:

- `test/test_corrmatrix.py` — split legacy `from
  multifunbrain.analysis.corrmatrix import ...` into canonical
  `from multifunbrain.io import load_correlation_matrix`,
  `from multifunbrain.preprocessing import detect_dead_regions,
  marchenko_pastur_denoise, prepare_correlation_matrix`, and
  `from multifunbrain.analysis.partition import adjusted_rand_index,
  compare_partition_sets`.
- `test/test_filtering.py` — `from multifunbrain.analysis.filtering
  import ...` → `from multifunbrain.processing import ...`.
- `test/test_netmetrics.py` — `from multifunbrain.analysis.netmetrics
  import ...` → `from multifunbrain.analysis.network import ...`.
- `notebooks/00_full_pipeline_demo.ipynb` — both import cells updated
  via `NotebookEdit` to canonical paths.
- `notebooks/multiscale_correlation_pipeline.ipynb` — same.

### Wildcard convention documented

`multifunbrain/notebook/__init__.py` gained a thorough docstring
explaining that the wildcard `from multifunbrain.X import *` pattern is
deliberate (single-line interactive ergonomic), should not be replaced
with explicit imports, and that the right hygiene investment is curated
`__all__` lists on the source modules. Cross-linked to the Ruff
per-file-ignores. (See user feedback memory
`feedback_notebook-module-wildcard.md`.)

### Docs

- `.claude/guide/module-map.md`: removed the "Legacy shim paths"
  section; added a "What changed in Phase 1" note plus an updated MP
  density row.
- `.claude/never-always/never.md`: removed the "Never drop the
  back-compat shim files" rule — it was honoured by this very PR
  sequence (Phase 0 retired the runtime imports; Phase 1 retired the
  shims themselves).
- Project memory (`MEMORY.md`): "Known cleanup tasks" entries for the
  Ruff config and shim removal moved into a "Resolved" section.

## Verification

- `ruff check multifunbrain/` — zero errors.
- `pytest test/` — 119 passed, same warnings as Phase 0.
- `python -c "from multifunbrain.analysis.lrglib import entropy"` →
  `ModuleNotFoundError` (as expected — the shim is gone).
- `python -c "from multifunbrain.core import hello_brain"` →
  `ModuleNotFoundError` (as expected — core.py is gone).
- `from multifunbrain.notebook import *` still works; wildcard
  namespace covers the curated `__all__` from `analysis/__init__.py`
  and `visualization/plotlib/__init__.py`.
- `multifunbrain hello Phase1` → `Hello, Phase1! Welcome to multifun-brain.`

## What's next

Phase 2 — crystallize plotting (style + monolith split):

- New `multifunbrain/visualization/style.py` with palettes,
  figsize defaults, rcParams helper.
- New `multifunbrain/visualization/plotlib/base.py` with
  `ensure_axes()` and `apply_decorations()`.
- Split the 1180-line `pipeline_plots.py` into
  `descriptive.py`, `filtering.py`, `network.py`, `lrg.py`,
  `grids.py`.
- Merge `lrg_multiscale_plots.py` into the new `lrg.py`.
- Merge `sankey_matplotlib.py` + `sankey_plotly.py` →
  `sankey.py` with `backend=` kwarg.
- Update `.claude/guide/plotting.md` with the templates and style
  module conventions.
