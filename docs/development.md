# Development guide

This page aggregates practical information for contributors hacking on
multifun-brain.

## Repository layout

```
multifunbrain/     # Source package
├── analysis/      # Graph-theoretic utilities
├── generation/    # Synthetic graph/time-series generators
├── visualization/ # Plotting helpers
├── cli.py         # Command-line entry point
└── core.py        # Shared utilities

docs/              # Markdown documentation served via MkDocs
notebooks/         # Exploratory Jupyter notebooks
test/              # Unit and integration tests
```

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

- Add type hints to new public functions.
- Keep functions small and composable; prefer helpers over long scripts.
- Document edge cases and parameter ranges in docstrings.
- Follow the standard library order for imports (stdlib, third-party,
  first-party).

## Documentation

- Update the Markdown files in `docs/` when features change.
- Include usage examples and references to notebooks if applicable.
- Consider adding diagrams or figures in `docs/assets/` when visual context
  helps users.

## Releasing

1. Update `CHANGELOG.md` with a new version section.
2. Bump the version in `pyproject.toml` and `multifunbrain/__init__.py`.
3. Tag the release: `git tag vX.Y.Z && git push --tags`.
4. Publish to PyPI: `python -m build && twine upload dist/*`.

## Support

File issues on GitHub or reach out via opensource@example.com.
