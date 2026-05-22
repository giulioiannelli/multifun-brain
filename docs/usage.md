# Usage guide

This guide walks through common workflows, from running the analysis pipeline on
correlation matrices to inspecting results and producing publication-ready plots.
All examples assume you have installed the package (see
[Installation](installation.md)).

## Correlation network analysis pipeline

The main feature is a three-section pipeline for signed correlation matrices from
fMRI data:

1. **Descriptive analysis** of the raw signed dense network (weight distribution,
   eigenvalue spectrum, Marchenko-Pastur validation, precision matrix, signed
   Laplacian, signed network metrics).
2. **Network filtering** via multiple methods that convert the signed matrix into
   unsigned tractable networks (absolute value, positive/negative split, backbone
   extraction, partial correlation).
3. **Standard network metrics + LRG multiscale** on each filtered network
   (global/node metrics, community detection, diffusion clustering).

### Command-line interface

The fastest way to run the pipeline:

```bash
# Single matrix
multifunbrain analyze data/correlation_matrices/my_matrix.pkl

# Entire directory (recursive, discovers .npy/.pkl/.csv/.txt files)
multifunbrain analyze data/correlation_matrices/

# Only Bold matrices, skip LRG (faster)
multifunbrain analyze data/correlation_matrices/ --pattern Bold --no-lrg

# With RMT validation (gamma = n_regions / n_timepoints)
multifunbrain analyze data/correlation_matrices/ --gamma 0.19

# Save to specific output directory
multifunbrain analyze data/correlation_matrices/ -o results/

# Preview which files will be processed without running anything
multifunbrain analyze data/correlation_matrices/ --list-only
```

Key options:

| Flag | Description |
|------|-------------|
| `--gamma FLOAT` | Aspect ratio p/n for RMT (enables MP validation) |
| `--filters METHOD [...]` | Filtering methods (default: `absolute positive negative`) |
| `--no-lrg` | Skip LRG multiscale analysis (much faster) |
| `--no-metrics` | Skip standard network metrics |
| `--pattern TEXT` | Only files with TEXT in the name (case-insensitive) |
| `--precision-method {direct,orie,graphical_lasso}` | Precision matrix method |
| `--no-recursive` | Do not search subdirectories |
| `--seed INT` | Random seed for reproducibility |
| `-o DIR` | Output directory (default: `./pipeline_results/`) |
| `--list-only` | List discovered files without running |

### Dead region handling

fMRI correlation matrices often contain **dead regions** — brain parcels whose
time series has zero variance, producing entire rows/columns of NaN in the
correlation matrix.  The pipeline automatically detects and **drops** these
regions instead of replacing NaN with zeros (which would fabricate non-existent
correlations).

- Dead regions are logged at runtime with their original indices.
- `PipelineResult.dropped_regions` records which indices were removed.
- `PipelineResult.n_regions_original` stores the size before dropping.
- The summary CSV/JSON include `n_regions_original` and `n_dropped` columns.

After dropping dead regions, any remaining sparse NaN/Inf entries (rare) are
replaced with 0 and a warning is logged.

### Output structure

```
pipeline_results/
  summary.csv              # one row per filter per matrix
  results.pkl              # full PipelineResult objects (reload with load_results)
  per_matrix/
    label__name.json       # JSON summary per matrix
```

### Python API

```python
from multifunbrain.pipeline import (
    PipelineConfig,
    run_pipeline,
    run_pipeline_directory,
    load_results,
)

# Run on a single matrix
result = run_pipeline(
    "data/correlation_matrices/my_matrix.pkl",
    config=PipelineConfig(
        gamma=0.19,
        filter_methods=["absolute", "positive"],
        run_lrg=False,
    ),
)

# Run on an entire directory tree
results = run_pipeline_directory(
    "data/correlation_matrices/",
    config=PipelineConfig(gamma=0.19, run_lrg=False),
    pattern="Bold",  # optional filename filter
)

# Access results
result.descriptive["weight_distribution"]["mean"]
result.descriptive["spectrum"]["n_signal"]
result.filtered_networks["absolute"]["graph"]  # nx.Graph
result.network_analyses["absolute"]["global_metrics"]["density"]
result.summary_table()  # DataFrame with key metrics
```

### Loading saved results

```python
from multifunbrain.pipeline import load_results

results = load_results("pipeline_results/")   # point at dir or results.pkl

results.labels                    # all labels
results[0]                        # by index
results["my_label"]               # by label
results.filter("Bold")            # subset by pattern
results.summary_table()           # combined DataFrame across all matrices
```

### Plotting

All plot functions accept an optional `ax=` parameter for composing custom
multi-panel figures, and return `(fig, ax)`.

```python
from multifunbrain.visualization import (
    plot_pipeline_summary,           # 6-panel overview
    plot_weight_distribution,        # positive/negative histogram
    plot_eigenvalue_spectrum,        # eigenvalues vs MP bounds
    plot_signed_laplacian_spectrum,  # signed Laplacian eigenmodes
    plot_correlation_matrix,         # diverging-colormap heatmap
    plot_filtered_comparison,        # bar chart across filters
    plot_node_metrics,               # per-node metric distributions
)

r = results[0]

# Full 6-panel summary
fig, axes = plot_pipeline_summary(r)
fig.savefig("summary.png", dpi=150)

# Individual plots
fig, ax = plot_weight_distribution(r)
fig, ax = plot_eigenvalue_spectrum(r)
fig, ax = plot_filtered_comparison(r, metric="modularity")
fig, axes = plot_node_metrics(r, filter_name="positive")

# Compose custom figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_weight_distribution(r, ax=ax1)
plot_eigenvalue_spectrum(r, ax=ax2)
```

### Notebook convenience

In Jupyter notebooks, import everything at once:

```python
from multifunbrain.notebook import *

results = load_results("pipeline_results/")
r = results[0]
plot_pipeline_summary(r)
```

---

## Synthetic data generation

### Command-line

```bash
multifunbrain generate-hmn --levels 3 --base-module-size 16 --p-in 0.8 --p-out 0.05 --seed 13 --output hmn.graphml
```

### Python

```python
from multifunbrain.generation import generate_hmn, generate_brain_timeseries

G = generate_hmn(levels=4, base_module_size=4, p_in=0.9, p_out=0.05, seed=7)
timeseries, time = generate_brain_timeseries(
    n_regions=G.number_of_nodes(),
    n_timepoints=1000,
    sampling_rate=250,
    return_time=True,
)
```

## Reproducible notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating end-to-end
workflows. The reference notebook is `00_full_pipeline_demo.ipynb`. Launch with:

```bash
jupyter lab
```

Deprecated notebooks are archived in `notebooks/archive/` (gitignored).

## Troubleshooting

- Ensure optional dependencies are installed when using Nilearn or Plotly-based
  features.
- If the `multifunbrain` command is not found, reinstall:
  `pip install -e .`
- If you encounter missing shared library errors, reinstall the Conda
  environment -- system-level dependencies are managed by conda-forge.
