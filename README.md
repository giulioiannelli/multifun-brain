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

### Analyse correlation matrices from the command line

```bash
# Single matrix
multifunbrain analyze data/correlation_matrices/my_matrix.pkl --no-lrg

# All matrices in a directory (recursive)
multifunbrain analyze data/correlation_matrices/ --pattern Bold --no-lrg

# With Marchenko-Pastur validation
multifunbrain analyze data/correlation_matrices/ --gamma 0.19 -o results/
```

### Or from Python / Jupyter

```python
from multifunbrain.pipeline import run_pipeline, load_results, PipelineConfig
from multifunbrain.visualization import plot_pipeline_summary

# Run pipeline on a single matrix
result = run_pipeline(
    "data/correlation_matrices/my_matrix.pkl",
    config=PipelineConfig(filter_methods=["absolute", "positive"], run_lrg=False),
)

# Load results from a previous CLI run
results = load_results("pipeline_results/")
results.summary_table()          # comparison DataFrame
r = results[0]
fig, axes = plot_pipeline_summary(r)  # 6-panel overview figure
```

The pipeline runs three analysis sections:

1. **Descriptive analysis** of the raw signed correlation network
2. **Network filtering** to produce unsigned tractable networks
3. **Standard metrics + LRG multiscale** on each filtered network

### Synthetic data generation

```bash
multifunbrain generate-hmn --levels 3 --base-module-size 8 --p-in 0.9 --p-out 0.05 --seed 42
```

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

## Interactive dashboard

A browser GUI lets non-coder collaborators explore analysis results from a single
localhost URL — no notebooks, no code. Minimal setup:

```bash
pip install -e ".[dashboard]"     # adds FastAPI + uvicorn (one-time)
./dashboard/run.sh                # builds the frontend once, serves http://localhost:8000
```

See [`dashboard/README.md`](dashboard/README.md) for how result bundles must be
placed (including the per-subject structure) and the full run/dev instructions.

## Project layout

```
multifunbrain/              # Installable Python package
├── analysis/               # Graph and signal processing utilities
│   ├── corrmatrix.py       #   I/O, matrix prep, denoising, LRG clustering
│   ├── descriptive.py      #   Signed network characterisation (Section 1)
│   ├── filtering.py        #   Network filtering / reduction (Section 2)
│   └── netmetrics.py       #   Standard network metrics (Section 3)
├── generation/             # Synthetic network and time-series generators
├── visualization/          # Matplotlib/Plotly-based helpers
│   └── plotlib/            #   Pipeline plots, Sankey, entropy, colorbars
├── pipeline.py             # Three-section pipeline orchestration
├── cli.py                  # Command-line interface (analyze, generate-hmn)
└── core.py                 # Shared core utilities

notebooks/                  # Jupyter notebooks (00_full_pipeline_demo.ipynb)
test/                       # Unit and integration tests (62 tests)
docs/                       # Markdown documentation rendered by MkDocs
data/                       # Correlation matrices from collaborators
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
