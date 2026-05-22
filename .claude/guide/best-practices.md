# Best practices

## Where things go

- **`multifunbrain/` package** — reusable, hierarchical, modular. Each
  function/class has *one* canonical home. Short, readable, no
  duplication. New utility? Find the scope-based subpackage that fits
  (`io`, `preprocessing`, `processing`, `analysis`, `datasets`,
  `pipeline`, `visualization`, `generation`). If none fits, add a new
  subpackage rather than dumping into the closest one.
- **`scripts/`** — thin orchestrators (≤80 lines each) that wire library
  calls into specific runs. One topic per subfolder
  (e.g. `scripts/april/`). Always include `argparse` + a guard
  `if __name__ == "__main__":` block.
- **`notebooks/`** — visual/narrative presentation only. Load
  pre-computed `results.pkl` via `load_results(...)`, plot, discuss. No
  analysis logic. If you need a new plot, add it to
  `multifunbrain/visualization/plotlib/` and call it from the notebook.

## Coding style

- `from __future__ import annotations` in every module. Use modern type
  syntax (`dict`, `list`, `X | Y`, `X | None`).
- Type-hint all new public functions and methods.
- Plot functions follow the signature: take an optional `ax=None`,
  return `(fig, ax)`.
- Keep functions small, composable, and explicit. Prefer wrappers and
  small helpers over long inline scripts.
- Ruff for lint, Black for formatting (88-char lines), pytest for tests.

## Adding a new function

1. Decide its scope (see "Where things go"). Place it in the matching
   subpackage.
2. Add it to the subpackage's `__init__.py` re-exports + `__all__`.
3. Write a test in `test/test_<topic>.py` (data-independent where possible).
4. If it's a public API addition, mention it in `docs/api_reference.md`.
5. Run `ruff check multifunbrain/` and `pytest test/`.

## Moving an existing function (refactor)

1. Create the new canonical home (file + `__init__.py` re-export).
2. Convert the old location to a re-export shim — keep the old import
   path working. Document the new home at the top of the shim file.
3. Run the full test suite — **all existing tests must still pass with
   zero modification**.
4. Add a `history/YYYY-MM-DD_<topic>.md` entry describing what moved
   and why.

## Branching + commits

- Feature work on a branch (e.g. `april-scaffold`). PR to `main` when verified.
- Commit messages: imperative, short subject + descriptive body
  explaining *why*.
- Never push to `main` directly. Never force-push to `main`.
- Don't commit large data files (the `data/*` rule in `.gitignore` covers
  it; double-check before `git add -A`).

## Notebook hygiene

- Self-contained: relative `Path` operations, no hard-coded user dirs.
- Markdown cells describe inputs/outputs of each section.
- Avoid heavy compute in notebooks — call into the library or a script,
  load results, plot.
- Archive deprecated notebooks under `notebooks/archive/` (gitignored).

## Reusing utilities

Before writing new code, search for an existing utility:

- I/O? `multifunbrain.io.{corrmatrix, results}`.
- Cleaning / dead regions / denoising? `multifunbrain.preprocessing.*`.
- Filtering / backbone / partial-corr? `multifunbrain.processing.*`.
- Descriptive / LRG / network metrics / partition compare?
  `multifunbrain.analysis.*`.
- Plotting? `multifunbrain.visualization.plotlib.*`.

See [`module-map.md`](module-map.md) for the full mapping.
