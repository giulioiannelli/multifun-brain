# Installation

multifun-brain is designed to be portable: install it with `pip`, use a Conda
environment, or integrate it inside an existing research workflow. This guide
covers each option with additional notes for GPU-accelerated use cases.

## Prerequisites

- Python **3.9** or newer.
- Basic scientific Python stack (NumPy, SciPy). These are installed
  automatically when using the default dependency set.

## Installing from PyPI

```bash
pip install multifunbrain
```

This command installs the core package with the standard runtime dependencies.
Heavy visualisation libraries are optional to keep the base install lightweight.

## Installing from source

1. Clone the repository:
   ```bash
   git clone https://github.com/giulioiannelli/multifun-brain.git
   cd multifun-brain
   ```
2. Install in editable mode together with the development extras:
   ```bash
   pip install -e .[dev]
   ```

Editable installs automatically pick up changes in the working tree.

## Optional dependency groups

| Extra | Description | Packages |
|-------|-------------|----------|
| `viz` | Interactive and publication-grade visualisation | Plotly, Matplotlib, Nilearn |
| `dev` | Tooling for formatting, linting, testing, and type checking | black, ruff, mypy, pytest, pytest-cov |
| `docs` | Documentation site generator | mkdocs, mkdocs-material |

Install extras using the standard bracket syntax, e.g.
`pip install "multifunbrain[viz,dev]"`.

## Conda environment

For users preferring Conda, an environment specification is bundled:

```bash
conda env create -f multifun-brain.yml
conda activate multifun-brain
```

The specification installs Python, the numerical stack, and an editable version
of the package with recommended extras.

## Verifying the installation

Run the automated tests to confirm the environment is healthy:

```bash
pytest
```

If you installed the optional type checker and linters, consider running them
as well:

```bash
ruff check .
black --check .
mypy multifunbrain
```

## Upgrading

```bash
pip install --upgrade multifunbrain
```

Check the [changelog](../CHANGELOG.md) for release notes before upgrading
production deployments.
