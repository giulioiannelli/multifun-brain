# API reference

This reference summarises the most commonly used functions in the
`multifunbrain` package. Inspect the source code for full signatures and
implementation details.

## `multifunbrain.core`

- `hello_brain(name: str) -> str` – Return a friendly greeting for quick sanity
  checks.
- `band_filter(data, low, high, fs=1.0, order=4, btype="bandpass") -> np.ndarray`
  – Apply a Butterworth filter along the last axis of the input array.
- `marchenko_pastur_density(eigenvalues, gamma, sigma=1.0) -> np.ndarray` –
  Evaluate the Marchenko–Pastur density for an array of eigenvalues.

## `multifunbrain.generation`

- `generate_hmn(levels=3, base_module_size=4, p_in=1.0, p_out=0.05, seed=None)` –
  Generate a hierarchical modular network.
- `generate_flower_graph(u=2, v=2, iterations=3)` – Construct a (u, v)-flower
  graph by recursively replacing edges.
- `generate_brain_timeseries(..., return_time=False)` – Multiple helper
  functions that synthesise multichannel time series with modular structure.

## `multifunbrain.analysis.corrnet`

- `compute_correlation_matrix(timeseries)` – Compute the Pearson correlation
  matrix of a multivariate time series.
- `marchenko_pastur_density(eigenvalues, gamma, sigma=1.0)` – Shortcut import of
  the core density helper for backwards compatibility.

## `multifunbrain.analysis.graphutils`

- `get_giant_component(graph, strongly=False)` – Return the largest connected
  component of a graph.
- `build_correlation_network(timeseries, regularize=True, ...)` – Construct a
  graph from correlation values with optional cleaning and thresholding steps.
- `compute_threshold_stats(graph)` – Sweep edge thresholds and compute giant
  component statistics.
- `compute_threshold_stats_fast(graph, n_points=0)` – Faster union–find powered
  version of the threshold sweep.

## `multifunbrain.analysis.lrglib`

- `rho_matrix(tau, L)` – Compute the diffusion kernel normalised to trace 1.
- `entropy(w, steps=600, t1=-2, t2=5)` – Diffusion-based spectral entropy.
- `compute_optimal_threshold(linkage_matrix, scaling_factor=1)` – Partition
  stability based threshold estimation.
- `identify_switching_nodes(partitions, tau_values)` – Track nodes that change
  community assignments across scales.

## `multifunbrain.visualization.plotlib`

- Collection of Matplotlib/Plotly-based plotting helpers such as
  `plot_degree_distribution` and `plot_matrix`.

For a complete, auto-generated reference consider integrating Sphinx or MkDocs
API plugins. The Markdown pages in this folder serve as a lightweight portable
baseline.
