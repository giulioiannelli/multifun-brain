# Development guide

This page aggregates practical information for contributors hacking on
multifun-brain.

## Repository layout

```
multifunbrain/                  # Library (one canonical home per function)
├── io/                         # load_correlation_matrix, load_results, ResultsCollection
├── preprocessing/              # dead-region detection, matrix prep, MP denoising
├── processing/                 # signed→unsigned filtering, backbones, partial corr, percolation
├── analysis/
│   ├── descriptive/            #   Section 1 — weights, spectrum, signed Laplacian
│   ├── lrg/                    #   Section 3 — LRG kernel, distance, partitions, compare
│   ├── network/                #   Section 3 — global/node/community metrics
│   ├── partition.py            #   ARI, partition-set comparison
│   └── emd_bands.py            #   EMD / IMF frequency bands
├── datasets/                   # dataset-specific loaders (april)
├── pipeline/                   # PipelineConfig, PipelineResult, runners, discovery
├── visualization/
│   ├── style.py                #   palettes, figsizes, rc profiles
│   └── plotlib/                #   descriptive / filtering / network / lrg / grids / sankey
├── generation/                 # synthetic-network generators
├── notebook/                   # wildcard re-exports for Jupyter
└── cli.py                      # command-line entry point

dashboard/                      # FastAPI backend + React/Vite frontend (collaborator GUI)
scripts/                        # thin orchestration scripts, one folder per topic
notebooks/                      # presentation notebooks (archive/ is gitignored)
test/                           # pytest suite
docs/                           # Markdown documentation (MkDocs)
data/                           # gitignored collaborator data + atlases
```

The function-level map lives in `.claude/guide/module-map.md`.

## Pipeline architecture

The analysis pipeline has three sections:

1. **Descriptive analysis** (`analysis/descriptive/`): weight distribution,
   eigenvalue spectrum, MP validation, precision matrix (direct / ORIE /
   GraphicalLasso), signed Laplacian, signed network metrics.
2. **Network filtering** (`processing/`): absolute threshold,
   positive/negative split, backbone extraction (disparity, LANS,
   MP-validated, percolation), partial correlation.
3. **Standard metrics + LRG** (`analysis/network/` + `analysis/lrg/`):
   global/node metrics, community detection, rich-club, diffusion clustering.

These are orchestrated by `multifunbrain/pipeline/` via `PipelineConfig` and
`PipelineResult` dataclasses. The CLI `analyze` subcommand wraps everything
for batch processing; the dashboard reads the resulting `results.pkl` files.

## Tooling

- **Formatting:** `black` (88 char line length).
- **Linting:** `ruff` for static analysis and import sorting.
- **Type checking:** `mypy` (non-strict by default).
- **Testing:** `pytest` with the configuration in `pyproject.toml`.

Run all checks before pushing:

```bash
ruff check .
black --check .
mypy multifunbrain
pytest
```

## Coding conventions

- Use `from __future__ import annotations` in all modules.
- Use modern type syntax (`dict`, `list`, `X | Y`, `X | None`) since all
  modules import future annotations.
- Add type hints to new public functions.
- Keep functions small and composable; prefer helpers over long scripts.
- Document edge cases and parameter ranges in docstrings.
- Follow the standard library order for imports (stdlib, third-party,
  first-party).
- Plot functions should accept an optional `ax=` parameter and return
  `(fig, ax)` for composability.

## Documentation

- Update the Markdown files in `docs/` when features change.
- Include usage examples and references to notebooks if applicable.
- The primary docs are: `usage.md` (workflows), `api_reference.md` (API),
  and this file (development).

## Releasing

1. Update `CHANGELOG.md` with a new version section.
2. Bump the version in `pyproject.toml` and `multifunbrain/__init__.py`.
3. Tag the release: `git tag vX.Y.Z && git push --tags`.
4. Publish to PyPI: `python -m build && twine upload dist/*`.

## Support

File issues on GitHub or reach out via opensource@example.com.
