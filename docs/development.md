# Development guide

This page aggregates practical information for contributors hacking on
multifun-brain.

## Repository layout

```
multifunbrain/                  # Source package
├── analysis/                   # Graph-theoretic utilities
│   ├── corrmatrix.py           #   I/O, matrix prep, denoising, LRG clustering
│   ├── corrnet.py              #   Correlation computation
│   ├── descriptive.py          #   Signed network characterisation (Section 1)
│   ├── filtering.py            #   Network filtering / reduction (Section 2)
│   ├── graphutils.py           #   Graph utilities and thresholding
│   ├── lrglib.py               #   Laplacian Renormalisation Group
│   └── netmetrics.py           #   Standard network metrics (Section 3)
├── generation/                 # Synthetic graph/time-series generators
├── visualization/              # Plotting helpers
│   └── plotlib/
│       ├── pipeline_plots.py   #   Plots for pipeline results
│       ├── entropy.py          #   Entropy / clustering coefficient
│       ├── sankey_matplotlib.py
│       ├── sankey_plotly.py
│       └── colorbars.py
├── notebook/                   # Convenience re-exports for Jupyter
├── pipeline.py                 # Three-section pipeline orchestration
├── cli.py                      # Command-line entry point
└── core.py                     # Shared utilities

notebooks/                      # Jupyter notebooks
├── 00_full_pipeline_demo.ipynb #   Reference demo
├── archive/                    #   Deprecated notebooks (gitignored)
└── ...
test/                           # Unit and integration tests
docs/                           # Markdown documentation (MkDocs)
data/                           # Correlation matrices from collaborators
```

## Pipeline architecture

The analysis pipeline has three sections:

1. **Descriptive analysis** (`analysis/descriptive.py`): weight distribution,
   eigenvalue spectrum, MP validation, precision matrix (direct / ORIE /
   GraphicalLasso), signed Laplacian, signed network metrics.
2. **Network filtering** (`analysis/filtering.py`): absolute threshold,
   positive/negative split, backbone extraction (disparity, LANS,
   MP-validated), partial correlation.
3. **Standard metrics + LRG** (`analysis/netmetrics.py` + `analysis/lrglib.py`):
   global/node metrics, community detection, rich-club, diffusion clustering.

These are orchestrated by `pipeline.py` via `PipelineConfig` and
`PipelineResult` dataclasses. The CLI `analyze` subcommand wraps everything
for batch processing.

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
