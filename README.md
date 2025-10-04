# multifun-brain

Tools for generating, analysing, and visualising hierarchical modular brain networks.
The package bundles synthetic network generators, correlation-network analysis
helpers, diffusion-based metrics, and lightweight visualisation utilities that can
be combined for computational experiments or reproducible demos.

## Why multifun-brain?

- **Research ready** – prototype multi-scale brain network experiments with
  batteries of graph metrics and hierarchical thresholding utilities.
- **Synthetic data included** – generate hierarchical modular graphs and
  realistic multichannel time series for testing new pipelines.
- **Completely installable** – ship the package as a `pip` installable wheel or a
  Conda environment with optional extras for notebooks, docs, and development.
- **Documented** – detailed usage guides and API references live in the
  [`docs/`](docs/index.md) directory to keep the codebase portable and easy to
  extend.

## Installation

### Stable release (PyPI or local wheel)

```bash
pip install multifunbrain
```

If you are working from a clone of this repository you can install the package
in editable mode:

```bash
pip install -e .
```

### Optional dependency groups

The heavy visualisation and documentation stacks are optional. Install them on
an as-needed basis:

```bash
# Interactive visualisation helpers (Plotly, Matplotlib, Nilearn)
pip install "multifunbrain[viz]"

# Development tooling (formatters, linters, type checkers, tests)
pip install "multifunbrain[dev]"

# Documentation toolchain (MkDocs with Material theme)
pip install "multifunbrain[docs]"
```

### Conda environment

A fully reproducible Conda environment is available in
[`multifun-brain.yml`](multifun-brain.yml):

```bash
conda env create -f multifun-brain.yml
conda activate multifun-brain
```

### Verification

Run the test-suite once everything is installed to ensure your environment is
healthy:

```bash
pytest
```

## Quick start

```python
import numpy as np

from multifunbrain.generation import generate_hmn
from multifunbrain.analysis import corrnet

# Build a hierarchical modular network with 3 levels of hierarchy
G = generate_hmn(levels=3, base_module_size=8, p_in=0.9, p_out=0.05, seed=42)

# Create a synthetic multichannel time series and compute pairwise correlations
signals = np.random.default_rng(42).normal(size=(G.number_of_nodes(), 500))
corr_matrix = corrnet.compute_correlation_matrix(signals)

# Extract the Marchenko-Pastur density for eigenvalue analysis
evals = np.linalg.eigvalsh(corr_matrix)
mp_density = corrnet.marchenko_pastur_density(evals, gamma=0.5)
```

Prefer the command line? Use the `multifunbrain` executable after installation:

```bash
multifunbrain generate-hmn --levels 3 --base-module-size 8 --p-in 0.9 --p-out 0.05 --seed 42
```

The command prints summary statistics to standard output and can optionally dump
the network to GraphML for further analysis.

## Documentation

The `docs/` folder contains portable Markdown documentation that can be rendered
with MkDocs (`pip install "multifunbrain[docs]"`). Key entry points:

- [`docs/index.md`](docs/index.md) – project overview and navigation hub.
- [`docs/installation.md`](docs/installation.md) – environment setup recipes.
- [`docs/usage.md`](docs/usage.md) – practical workflows and CLI usage.
- [`docs/api_reference.md`](docs/api_reference.md) – module level API summaries.
- [`docs/development.md`](docs/development.md) – contributing guidelines,
  testing, releasing, and code-style expectations.

To preview the documentation site locally:

```bash
mkdocs serve
```

## Project layout

```
multifunbrain/             # Installable Python package
├── analysis/              # Graph and signal processing utilities
├── generation/            # Synthetic network and time-series generators
├── visualization/         # Matplotlib/Plotly-based helpers
├── cli.py                 # Portable command-line interface entry point
├── core.py                # Shared core utilities
└── py.typed               # Enables type checking for consumers

notebooks/                 # Example Jupyter notebooks
test/                      # Legacy exploratory scripts kept for parity
docs/                      # Markdown documentation rendered by MkDocs
```

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on our preferred
workflow, coding conventions, and development tooling. By participating in this
project you agree to abide by the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Bug reports and feature requests are tracked through GitHub issues. Pull
requests are welcome—lint, type-check, and test before submitting.

## License

multifun-brain is released under the [MIT License](LICENSE).

## Citation

If you use this project in academic work, please cite it as:

> Multifun-Brain Developers. (2024). *multifun-brain* (Version 0.2.0) [Computer
> software]. https://github.com/your-org/multifun-brain

## Acknowledgements

This project builds upon open-source contributions from the NetworkX, NumPy,
SciPy, Plotly, and Nilearn communities.
