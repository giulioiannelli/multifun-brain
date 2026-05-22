# Plotting & visualization

Every plot pattern is defined **once** in
`multifunbrain.visualization.plotlib.*` and is reused from there.
Notebooks, scripts, and reports all call those helpers — they do not
re-derive plot logic inline. This guide is the rule book.

## Where plot code lives

Each section of the pipeline has its own file under
`multifunbrain/visualization/plotlib/`:

- **`descriptive.py`** — Section 1 plots: `plot_weight_distribution`,
  `plot_eigenvalue_spectrum`, `plot_signed_laplacian_spectrum`,
  `plot_signed_balance`, `plot_correlation_matrix`.
- **`filtering.py`** — Section 2 plots: `plot_percolation_curve`,
  `plot_filtered_comparison`.
- **`network.py`** — Section 3 standard-network plots: `plot_node_metrics`,
  `plot_signed_network`, `plot_network`.
- **`lrg.py`** — Section 3 LRG plots (both per-`PipelineResult` and
  per-`MultiscaleResult`): `plot_lrg_entropy`, `plot_lrg_dendrogram`,
  `plot_lrg_psi`, `plot_lrg_partition_network`, `plot_lrg_sankey`,
  `plot_specific_heat`, `plot_specific_heat_overlay`,
  `plot_dendrogram_with_psi`, `plot_psi_curve`, `plot_rmi_curve`,
  `plot_partition_flow`, `plot_tanglegram`, `plot_metastable_overlay`.
- **`grids.py`** — `plot_results_grid` (canonical multi-panel composer)
  and `plot_pipeline_summary` (six-panel single-result overview).
- **`sankey.py`** — `plot_sankey(..., backend="plotly"|"matplotlib")`
  merging both backends behind one entry point.
- **`entropy.py`** — atomic helper `plot_entropy_and_C(ax, t, Sm1, dS)`
  for the dual-axis LRG curves.
- **`colorbars.py`** — colorbar utilities (e.g.
  `imshow_colorbar_caxdivider`).
- **`base.py`** — shared scaffolding (`ensure_axes`,
  `apply_decorations`) every helper builds on.
- **`_helpers.py`** — package-private helpers used by multiple sections
  (`_get_section`, `_resolve_filter`, `_signed_laplacian_embedding`,
  `_network_layout`, `_nearest_tau_index`).

Style defaults — palettes, figure sizes, rcParams profiles — live one
level up at `multifunbrain/visualization/style.py`. Pull colours and
sizes from there instead of hardcoding hex strings.

If a plot pattern is repeated in two places, promote it into the
section file that fits before the third copy exists.

## Templates and the style module

`multifunbrain/visualization/style.py` is the single source of truth
for colours and sizes. The named tables are:

| Constant | Use |
|---|---|
| `PALETTES["material"]` | Material Design accent palette — the project default. Keys: `primary`, `primary_dark`, `primary_light`, `accent`, `accent_light`, `danger`, `danger_light`, `success`, `success_light`, `neutral`. |
| `PALETTES["spectral"]` | 13-colour qualitative palette for partition / module colouring. |
| `PALETTES["signed"]` | `{"pos": ..., "neg": ...}` for signed-edge networks. |
| `PALETTES["contrast"]` | `{"co2": ..., "rest": ...}` for April-batch contrast comparisons. |
| `FIGSIZE` | Standard figure sizes (`single`, `wide`, `tall`, `square`, `grid_cell`, `double`). |
| `rc_context(profile)` | Context manager for one-shot rcParams profiles (`paper`, `slide`). |

### Crystallized templates

When the user says "dendrogram", they mean **the** dendrogram template.
Each template below is a canonical recipe: signature + which constants
to import + the one-liner that produces a project-consistent figure.

- **`DENDROGRAM`** — `multifunbrain/visualization/plotlib/lrg.py::plot_lrg_dendrogram`. Threshold line uses `PALETTES["material"]["danger_light"]`. Wide figsize.
- **`CORRELATION_HEATMAP`** — `descriptive.py::plot_correlation_matrix`. `FIGSIZE["square"]`, diverging `RdBu_r` cmap, `TwoSlopeNorm` centred at zero.
- **`PSI_CURVE`** — `lrg.py::plot_psi_curve`. `FIGSIZE["single"]`; cluster-marker uses Material `danger`.
- **`SPECIFIC_HEAT`** — `lrg.py::plot_specific_heat`. Trace uses Material `primary_dark`; τ′ uses `accent`; τ* uses `success`.
- **`SANKEY_FLOW`** — `sankey.py::plot_sankey(partitions, taus, backend="plotly")` for interactive view; `backend="matplotlib"` for static PDF.
- **`RESULTS_GRID`** — `grids.py::plot_results_grid` is the *only* approved way to render N results side-by-side. Never write a `for r in results: plt.subplots(...)` loop in a notebook.

When adding a new template:

1. Pick the section file that owns the concept (`descriptive`,
   `filtering`, `network`, `lrg`, `grids`, `sankey`).
2. Use `ensure_axes(ax, figsize=FIGSIZE["..."])` for the create-or-reuse
   boilerplate.
3. Pull every colour from `PALETTES` — no inline hex strings.
4. Accept the standard decoration kwargs (`title`, `colorbar`,
   `legend`) so the function composes with `plot_results_grid`.
5. Export from the section file's `__all__` and add a row to the
   template table above.

## Signature convention

All public plot functions follow the same shape:

```python
def plot_xxx(
    result,                            # PipelineResult | sub-dict | ndarray
    *,
    ax: Axes | None = None,            # composes into a parent figure
    # ...optional decorations as kwargs (see below)...
) -> tuple[Figure, Axes]:
    ...
    if ax is None:
        fig, ax = plt.subplots(figsize=...)
    else:
        fig = ax.get_figure()
    # plot
    return fig, ax
```

Decoration kwargs that callers may want to suppress when composing
into a grid:

- `title: str | None` — pass `None` (or `title=None`) to skip.
- `colorbar: bool = True` — pass `False` to skip the per-cell colorbar
  (the grid usually adds one shared colorbar instead).
- `legend: bool = True` — pass `False` for tiny grid cells.

If a new plot function adds a colorbar or legend by default, **also
add the corresponding bool kwarg** so it stays grid-friendly.

## Multi-panel layouts — use `plot_results_grid`

For "show the same plot for N results in a grid", the canonical helper
is `multifunbrain.visualization.plotlib.plot_results_grid`:

```python
from multifunbrain.visualization.plotlib.pipeline_plots import (
    plot_correlation_matrix,
    plot_results_grid,
)

fig, _ = plot_results_grid(
    results_by_key,          # dict[(row_key, col_key) -> PipelineResult]
    plot_correlation_matrix, # any plot_fn(result, *, ax, **kwargs)
    row_keys=['co2', 'rest'],
    col_keys=['bpfBOLD', 'bpfVASO', 'MIRNoise_bold', 'optcom_bold', 'optcomMIRDenoised_bold'],
    figsize_per_cell=(2.5, 2.5),
    suptitle='Prepared correlation matrices',
    colorbar=False,          # forwarded to plot_fn
    vmax=1.0,
)
```

Behaviour:

- Missing keys → an empty "no data" cell.
- Per-cell exceptions → the error string is rendered in the cell;
  one bad subplot doesn't abort the figure.
- Per-cell row/col labels are placed automatically (leftmost column +
  top row).

**Never** inline a `for r in results: fig, ax = plt.subplots(...)` loop
in a notebook. Use `plot_results_grid` (or write a wrapper that does).

## Notebook style

A presentation notebook section is **3 cells maximum**:

1. Markdown title + one-line description.
2. A single call to a plot helper (most often `plot_results_grid`).
3. `plt.show()` only if the cell does not already return the figure.

If you find yourself writing nested loops, fixture variables, or
extracting partition data inside a notebook cell — stop. That belongs
in the library.

## Adding a new plot

1. Identify the section file that owns the concept (`descriptive`,
   `filtering`, `network`, `lrg`, `grids`, `sankey`). Grep that file
   to confirm no existing helper covers it.
2. Write the new helper with the standard signature
   (`result, *, ax=None, **decoration_kwargs`) using `ensure_axes`
   from `base.py` to handle the axes boilerplate.
3. Pull colours from `multifunbrain.visualization.style.PALETTES`
   and sizes from `FIGSIZE`. Do not inline hex strings.
4. Add the function to the section file's `__all__`.
5. Re-export it from
   `multifunbrain/visualization/plotlib/__init__.py` (`from .<section>
   import ...` block + `__all__`).
6. If it's grid-composable, expose `title`, `colorbar`, `legend`
   kwargs (use `apply_decorations` from `base.py`).
7. Use it from the notebook in 1–3 lines.
8. If the plot is a *new* crystallized template, add a row to the
   "Crystallized templates" table above.

## Verifying a notebook before handoff

Before saying "open notebook X" the assistant must execute every cell
headlessly and confirm zero `error` outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/april/01_global_overview.ipynb
python3 -c "
import json
nb = json.load(open('notebooks/april/01_global_overview.ipynb'))
errs = [(i, o['ename'], o['evalue'])
        for i, c in enumerate(nb['cells']) if c['cell_type']=='code'
        for o in c.get('outputs', []) if o.get('output_type')=='error']
print('clean' if not errs else errs)
"
```

If any cell errors, fix the underlying code (library or notebook),
re-execute, and verify clean output. Same rule for scripts: run them
locally before declaring them ready. This is enforced by
`.claude/never-always/never.md` ("Never tell the user to run a
notebook or script you haven't executed").

## When a notebook depends on data that may not exist yet

Guard the load cell so the notebook executes cleanly even when the
upstream pipeline hasn't run:

```python
RESULTS_DIR = Path('../../data/correlation_matrices_results/april/bands')
if (RESULTS_DIR / 'results.pkl').exists():
    results = load_results(RESULTS_DIR)
else:
    results = None
    print(f'No results at {RESULTS_DIR}. Run scripts/april/02_run_bands.py first.')
```

Subsequent cells short-circuit when `results is None`. The notebook
becomes self-documenting about its prerequisites.

## Cross-references

- `.claude/never-always/never.md` — "never hand off a notebook without
  executing it"; "never inline analysis logic in a notebook".
- `.claude/guide/pipeline-usage.md` — how to obtain `PipelineResult`
  objects.
- `.claude/guide/module-map.md` — full mapping of plot helpers to
  their submodules.
