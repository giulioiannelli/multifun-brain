# Usage guide

This guide walks through common workflows, from generating synthetic data to
running analyses and visualisations. All examples assume you have installed the
package (see [Installation](installation.md)).

## Command-line interface

The package installs a `multifunbrain` console script. See its help page:

```bash
multifunbrain --help
```

### Generate a hierarchical modular network

```bash
multifunbrain generate-hmn --levels 3 --base-module-size 16 --p-in 0.8 --p-out 0.05 --seed 13 --output hmn.graphml
```

Arguments:

- `--levels`: number of hierarchical levels.
- `--base-module-size`: size of the leaf modules.
- `--p-in`: intra-module connection probability.
- `--p-out`: inter-module connection probability.
- `--seed`: random seed (optional).
- `--output`: save the generated graph as GraphML (optional).

The command prints summary statistics to standard output and writes the graph if
`--output` is provided.

## Python API examples

### Generating data

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

### Building a correlation network

```python
from multifunbrain.analysis import corrnet, graphutils

corr_matrix = corrnet.compute_correlation_matrix(timeseries)
G_corr, removed = graphutils.build_correlation_network(corr_matrix)
print(f"Removed {len(removed)} isolated nodes")
```

### Threshold selection

```python
from multifunbrain.analysis.graphutils import compute_threshold_stats
from multifunbrain.analysis.lrglib import compute_optimal_threshold

thresholds, edges_frac, nodes_frac = compute_threshold_stats(G_corr)
flat_threshold, *_ = compute_optimal_threshold(thresholds[:, None])
```

### Visualising results

```python
from multifunbrain.visualization import plotlib

plotlib.plot_degree_distribution(G)
```

## Reproducible notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating end-to-end
analysis pipelines. Launch them with:

```bash
jupyter lab
```

## Exporting results

- Use NetworkX writers (`nx.write_graphml`, `nx.write_gpickle`) to persist graphs.
- Save time-series data as NumPy arrays (`np.save`) or Pandas DataFrames.

## Troubleshooting

- Ensure optional dependencies are installed when using Nilearn or Plotly-based
  features.
- If you encounter missing shared library errors, reinstall the Conda
  environment—system-level dependencies (e.g., `libstdc++`) are managed by conda-forge.
