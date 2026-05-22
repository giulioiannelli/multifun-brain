# Plotting & visualization

Every plot pattern is defined **once** in
`multifunbrain.visualization.plotlib.*` and is reused from there.
Notebooks, scripts, and reports all call those helpers — they do not
re-derive plot logic inline. This guide is the rule book.

## Where plot code lives

- `multifunbrain/visualization/plotlib/pipeline_plots.py` — the main
  family of plots that take a `PipelineResult` (or a sub-dict /
  ndarray) and produce a figure.
- `multifunbrain/visualization/plotlib/entropy.py` — atomic helper
  `plot_entropy_and_C(ax, t, Sm1, dS)` for the dual-axis LRG curves.
- `multifunbrain/visualization/plotlib/sankey_matplotlib.py`,
  `.../sankey_plotly.py` — Sankey backends.
- `multifunbrain/visualization/plotlib/colorbars.py` — colorbar
  utilities (e.g. `imshow_colorbar_caxdivider`).

If a plot pattern is repeated in two places, it belongs in
`pipeline_plots.py`. Promote it before the third copy exists.

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

1. Grep `pipeline_plots.py` to confirm no existing helper covers it.
2. Write the new helper in `pipeline_plots.py` with the standard
   signature (`result, *, ax=None, **decoration_kwargs`).
3. Add it to the module's `__all__`.
4. Re-export it from
   `multifunbrain/visualization/plotlib/__init__.py` (`from .pipeline_plots import ...` block + `__all__`).
5. If it's grid-composable, expose `title`, `colorbar`, `legend`
   kwargs.
6. Use it from the notebook in 1–3 lines.

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
