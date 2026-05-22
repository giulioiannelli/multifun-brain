# 2026-05-20 — Initial state (pre-reorg snapshot)

## Snapshot summary

Captured at the start of the cleaning-repository session, immediately
before the package reorganization (Part 1 of the plan
`/home/opisthofulax/.claude/plans/we-need-to-make-luminous-scone.md`).

## Package layout (flat)

```
multifunbrain/
├── __init__.py          — re-exports pipeline + core API
├── __main__.py
├── cli.py               — argparse-based CLI (~648 lines, many subcommands)
├── core.py              — band_filter, hello_brain, marchenko_pastur_density
├── pipeline.py          — PipelineConfig, PipelineResult, ResultsCollection, runners (all in one file)
├── analysis/
│   ├── __init__.py      — re-exports everything below
│   ├── corrmatrix.py    — I/O + preprocessing + LRG partitions + ARI (mixed concerns)
│   ├── corrnet.py       — Pearson + MP density wrappers
│   ├── descriptive.py   — weights / spectrum / signed Laplacian / precision / report
│   ├── filtering.py     — absolute, split-sign, validated (disparity/LANS/MP), partial-corr, apply_all
│   ├── graphutils.py    — giant component, threshold heuristics
│   ├── lrglib.py        — graph Laplacian, rho_matrix, entropy, partition helpers
│   └── netmetrics.py    — global / node / community / degree / rich-club / summary
├── datasets/            — (empty, present for placeholder)
├── generation/          — synthetic-network generators
├── notebook/            — Jupyter convenience re-exports
└── visualization/
    └── plotlib/
        ├── pipeline_plots.py
        ├── entropy.py
        ├── sankey_*.py
        └── colorbars.py

test/                    — 62 unit tests across 5 modules (all passing)
notebooks/               — 19 notebooks (some in archive/)
docs/                    — index, installation, usage, api_reference, development, faq
data/                    — gitignored; collaborator data
```

## Tests passing

`pytest test/ -q` → 62 passed.

## Known issues / blurriness

1. **Module boundaries blurred.** `analysis/corrmatrix.py` mixes I/O,
   preprocessing, LRG, and partition-comparison concerns. `descriptive.py`
   lumps weights / spectrum / signed-Laplacian / precision-matrix together.
   `pipeline.py` mixes config, result, runner, discovery, loader.
2. **`.claude/` directory near-empty.** Only `settings.local.json`. No
   guidance for agents on layout, rules, or evolution.
3. **`AGENTS.md` thin** (16 lines, generic). Useful but not indexed.
4. **`pyproject.toml` ruff config uses deprecated top-level keys** (need
   to migrate to `[tool.ruff.lint]` section) — out of scope today.
5. **82 pre-existing ruff lints** in `multifunbrain/`. Mostly F401
   (unused imports in re-export `__init__.py`s), UP035 (deprecated
   typing imports), I001 (import order). Out of scope today.

## Active data context

Daniele (collaborator) dropped a new correlation-matrix batch at
`data/correlation_mat_april_data/` on 2026-04-23 — 169 MB, 226 pickles,
stratified by contrast (co2/rest) × processing variant (5) × band
(s4/s5/sstar), with three aggregation levels (global / per-band / per-
subject×band, 6 subjects). The repo has no loader or orchestration for
this dataset.

## Where this entry sits

This is the *initial* state of the repo at the start of the cleaning-
repository session. The next history entries
(`2026-05-20_package-reorganization.md`,
`2026-05-20_april-scaffold-init.md`) describe what was changed.
