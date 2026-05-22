# 2026-05-22 — Phase 0: hygiene & safety net

First phase of the multi-phase cleanup described in
`/home/opisthofulax/.claude/plans/prologue-the-scope-hashed-honey.md`.
Pure hygiene — no semantic behavior changes. Branch: `phase-0-hygiene`.

## What changed

### Git hygiene

- `.gitignore` expanded from 7 lines to cover all standard Python tooling
  caches (`.ruff_cache/`, `.pytest_cache/`, `.mypy_cache/`, `*.egg-info/`,
  `.coverage`, `htmlcov/`, `.tox/`, `build/`, `dist/`, `.eggs/`) plus
  editor/OS noise (`.vscode/`, `.DS_Store`, `*.swp`, `*.swo`),
  `.env`/`.envrc`, `.tmp/`, and the post-Phase-6 `labs/archive/` path.
- Untracked the seven files that had slipped in:
  `git rm --cached .vscode/settings.json multifunbrain.egg-info/*`.
- Deleted `test/hmn.py` (125 L) and `test/hmn_2.py` (16 L) — orphaned
  non-test code; `multifunbrain.generation.generators.generate_hmn`
  supersedes them and nothing imported the originals.

### Ruff config + lint cleanup

- Migrated `[tool.ruff] select/ignore` → `[tool.ruff.lint] select/ignore`
  (deprecated since Ruff 0.5). Added `[tool.ruff.lint.per-file-ignores]`
  for `**/__init__.py` (F401 — re-exports) and `notebook/__init__.py`
  (F401/F403/E402 — deliberate wildcard convenience namespace).
- Applied `ruff check --fix --unsafe-fixes` for typing modernization
  (`typing.List/Tuple` → `list/tuple`, `typing.Sequence` →
  `collections.abc.Sequence`) and import ordering. Safe under
  `from __future__ import annotations` (always-rule).
- Manual fixes:
  - `multifunbrain/cli.py`: moved `log = logging.getLogger(__name__)`
    below the imports (E402).
  - `multifunbrain/generation/generators.py`: renamed `l` → `n_level`
    in HMN builder (E741).
  - `multifunbrain/analysis/graphutils.py`: added `# noqa: B023` to the
    Union-Find inner-closure lines in `compute_threshold_stats_fast`.
    The closure-over-loop-variable pattern is intentional (each loop
    iteration rebuilds `parent`/`rank`/`component_size` from scratch);
    proper refactor into a helper class is deferred.
- `ruff check multifunbrain/` now reports zero errors.

### Targeted fixes

- Removed two stray `print()` debug statements in
  `multifunbrain/analysis/graphutils.py`
  (`compute_optimal_threshold_std`).
- Migrated the three runtime `from ...analysis.lrglib import ...` imports
  in `multifunbrain/visualization/plotlib/pipeline_plots.py:992,993,1072`
  to canonical paths (`...analysis.lrg.kernel` and
  `...analysis.lrg.partitions`). Also updated the matching docstring at
  line 986. The flat aggregator `analysis/__init__.py` still imports
  via the shims — that's Phase 1's job.

### Tooling

- Added `.pre-commit-config.yaml` with: `pre-commit-hooks` (whitespace,
  EOL, YAML/TOML syntax, merge-conflict markers, large-file guard at
  1 MB, private-key detection); `ruff` (check + format); `nbstripout`
  for notebooks.
- Added `.github/workflows/ci.yml` — Python 3.10/3.11/3.12 matrix
  running `pip install -e .[dev]` → `ruff check` → `pytest test/ -ra`
  on push/PR to `main`.

### Docs

- Shrunk `AGENTS.md` from 25 lines to a 4-line redirect to `CLAUDE.md`.
- Created `.claude/CLAUDE.md` as a thin pointer to root `CLAUDE.md`, so
  the existing "treat the two as the same document" wording in the root
  file matches the filesystem.

## Verification

- `ruff check multifunbrain/` — zero errors.
- `pytest test/` — 119 passed, 7 warnings (pre-existing
  `RuntimeWarning` from skew/kurtosis on identical data; not Phase 0).
- `python -c "import multifunbrain; from multifunbrain.notebook import *"` —
  clean.
- `multifunbrain hello Phase0` → `Hello, Phase0! Welcome to multifun-brain.`

## What's next

Phase 1 — retire the analysis shims:
- Delete `multifunbrain/analysis/{corrmatrix,filtering,netmetrics,lrglib}.py`.
- Rewrite `analysis/__init__.py` to import only from canonical homes.
- Collapse `multifunbrain/core.py` (move `marchenko_pastur_density` →
  `preprocessing/denoising.py`; drop `hello_brain` into the CLI;
  delete the file).
- Add a docstring at the top of `notebook/__init__.py` documenting the
  deliberate wildcard pattern so future agents don't try to "fix" it.
- Drop the "shims must not be removed without a migration PR" rule from
  `.claude/never-always/never.md` once those deletions land (it was
  honoured by this very PR sequence).
