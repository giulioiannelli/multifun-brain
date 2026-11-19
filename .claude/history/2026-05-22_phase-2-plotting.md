# 2026-05-22 — Phase 2: crystallize plotting (style + section split)

Third phase of the multi-phase cleanup described in
`/home/opisthofulax/.claude/plans/prologue-the-scope-hashed-honey.md`.
Branch: `phase-2-plotting-templates` (off `main` after PR #6 and PR #8
landed earlier today).

## What changed

### New foundation files

- **`multifunbrain/visualization/style.py`** (new, ~130 L): single
  source of truth for plot styling. `PALETTES["material"]` (10-colour
  Material palette), `PALETTES["spectral"]` (13-colour qualitative),
  `PALETTES["signed"]` (`pos`/`neg`), `PALETTES["contrast"]`
  (`co2`/`rest`). `FIGSIZE` (`single`/`wide`/`tall`/`square`/`grid_cell`/`double`).
  `rc_context(profile)` context manager for `paper` / `slide` rcParams.
- **`multifunbrain/visualization/plotlib/base.py`** (new, ~95 L):
  `ensure_axes(ax, figsize=..., **subplots_kwargs)` to handle the
  if-None-create-else-get_figure boilerplate; `apply_decorations(ax,
  title=, xlabel=, ylabel=, legend=, grid=, ...)` for the standard
  decoration kwargs every plot helper accepts.
- **`multifunbrain/visualization/plotlib/_helpers.py`** (new, ~115 L):
  private helpers used across section files: `_get_section` (extract
  a descriptive sub-dict from a `PipelineResult` or raw dict),
  `_resolve_filter` (default to first available), `_signed_laplacian_embedding`
  (2-D spectral embedding for signed networks), `_network_layout`
  (Kamada–Kawai or spring), `_nearest_tau_index` (snap to τ grid).

### Section split

The 1180-line `pipeline_plots.py` and the 428-line
`lrg_multiscale_plots.py` were carved into per-concept files:

- **`descriptive.py`** (~260 L): `plot_weight_distribution`,
  `plot_eigenvalue_spectrum`, `plot_signed_laplacian_spectrum`,
  `plot_signed_balance`, `plot_correlation_matrix`.
- **`filtering.py`** (~225 L): `plot_percolation_curve`,
  `plot_filtered_comparison`.
- **`network.py`** (~210 L): `plot_node_metrics`,
  `plot_signed_network`, `plot_network`.
- **`lrg.py`** (~625 L): all 13 LRG plots (per-`PipelineResult` and
  per-`MultiscaleResult`), merging the 5 `plot_lrg_*` from the old
  `pipeline_plots.py` with all 8 from `lrg_multiscale_plots.py`.
- **`grids.py`** (~210 L): `plot_results_grid` (canonical multi-panel
  composer) + `plot_pipeline_summary` (six-panel single-result
  overview).
- **`sankey.py`** (~225 L): merged matplotlib + Plotly backends behind
  one entry point `plot_sankey(partitions, taus, backend="plotly"|"matplotlib")`.
  `plot_sankey_matplotlib` is preserved as a thin back-compat alias.

### Deletions

- `multifunbrain/visualization/plotlib/pipeline_plots.py` (1180 L).
- `multifunbrain/visualization/plotlib/lrg_multiscale_plots.py` (428 L).
- `multifunbrain/visualization/plotlib/sankey_matplotlib.py` (137 L).
- `multifunbrain/visualization/plotlib/sankey_plotly.py` (63 L).

Public API preserved: `__all__` in
`multifunbrain/visualization/plotlib/__init__.py` covers every symbol
exported by the old monoliths, plus the new ones (`ensure_axes`,
`apply_decorations`).

### Crystallized templates (initial set)

Three representative plots refactored to pull from `style.PALETTES` and
`FIGSIZE` instead of inline hex strings — proving the pattern future
plots will follow:

- **`CORRELATION_HEATMAP`** (`descriptive.py::plot_correlation_matrix`):
  `FIGSIZE["square"]` for default canvas.
- **`DENDROGRAM`** (`lrg.py::plot_lrg_dendrogram`): threshold line
  colour pulled from `PALETTES["material"]["danger_light"]`; wide
  figsize.
- **`SPECIFIC_HEAT`** (`lrg.py::plot_specific_heat`): trace from
  `PALETTES["material"]["primary_dark"]`; τ′ line from `accent`; τ*
  line from `success`.

The remaining ~25 plots keep their original hardcoded colours for now;
follow-up PRs can convert them as the user reviews each pattern.

### Docs

- `.claude/guide/plotting.md` rewritten to reflect the section-file
  layout, document `style.py` constants, list the canonical templates
  (`DENDROGRAM`, `CORRELATION_HEATMAP`, `PSI_CURVE`, `SPECIFIC_HEAT`,
  `SANKEY_FLOW`, `RESULTS_GRID`), and update the "Adding a new plot"
  recipe to mention `ensure_axes` + `PALETTES` + `FIGSIZE`.

## Verification

- `ruff check multifunbrain/` — zero errors.
- `pytest test/` — 119 passed.
- `from multifunbrain.visualization.plotlib import *` — succeeds with
  full `__all__` exposed.
- `jupyter nbconvert --to notebook --execute --inplace
  notebooks/00_full_pipeline_demo.ipynb` — clean, zero error outputs.

## What's next

Phase 3 — CLI decomposition. The 758-line `multifunbrain/cli.py` will
be split into `multifunbrain/cli/` (parser.py, plot_registry.py,
commands/{analyze,plot,compare_partitions,results_to_csv,generate_hmn,hello}.py).
The plot registry will be made declarative so adding a plot doesn't
require editing the CLI dispatcher.
