# Contributing to multifun-brain

Thank you for considering a contribution! This document summarises the workflow
and expectations for collaborators.

## Getting started

1. **Fork and clone** the repository.
2. Create and activate the recommended environment:
   ```bash
   conda env create -f multifun-brain.yml
   conda activate multifun-brain
   ```
   or install the package locally with development extras:
   ```bash
   pip install -e .[dev]
   ```
3. Install optional extras relevant to your contribution (`[viz]`, `[docs]`).

## Development workflow

1. Create a feature branch: `git checkout -b feat/my-feature`.
2. Make your changes, ensuring that code is documented and type-annotated.
3. Format and lint:
   ```bash
   ruff check .
   black .
   mypy multifunbrain
   ```
4. Run the tests: `pytest`.
5. Commit with a descriptive message and open a pull request.

## Documentation standards

- Every public function should contain a docstring explaining arguments and
  return values.
- Update `docs/` when behaviour changes.
- Screenshots or rendered notebooks should live in the `docs/assets/` folder (git
  LFS is recommended for large binaries).

## Pull request checklist

- [ ] Tests passing (`pytest`).
- [ ] Linting and formatting applied.
- [ ] Updated documentation and changelog entry.
- [ ] Linked issues referenced in the PR description.

By contributing you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
