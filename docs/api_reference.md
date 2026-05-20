# API reference

This reference summarises the public functions in `multifunbrain`. Inspect the
source code for full signatures and implementation details.

## `multifunbrain.pipeline`

The main entry points for running analyses on correlation matrices.

### Configuration and results

- `PipelineConfig` -- dataclass with all pipeline settings (gamma, filter_methods,
  precision_method, run_lrg, seed, etc.). All fields have sensible defaults.
- `PipelineResult` -- container for all outputs: `.descriptive`, `.filtered_networks`,
  `.network_analyses`, `.lrg_results`, `.corr_prepared`. Has `.to_dict()` and
  `.summary_table()` methods.
- `ResultsCollection` -- list-like container returned by `load_results()`.
  Supports integer indexing, label-based lookup (`results["label"]`),
  `.filter("pattern")`, `.labels`, and `.summary_table()`.

### Functions

- `run_pipeline(corr, config=None, label=None) -> PipelineResult` -- run the
  full three-section pipeline on a single matrix (array or file path).
- `run_pipeline_batch(inputs, config=None, labels=None) -> list[PipelineResult]`
  -- run the pipeline on a list of matrices.
- `run_pipeline_directory(directory, config=None, recursive=True, pattern=None) -> list[PipelineResult]`
  -- discover all matrix files in a directory tree and run the pipeline on each.
- `discover_matrices(directory, extensions=None, recursive=True, pattern=None) -> list[Path]`
  -- find matrix files without running the pipeline.
- `load_results(path) -> ResultsCollection` -- load results from a previous
  `multifunbrain analyze` run (accepts a directory or `results.pkl` path).

---

## `multifunbrain.analysis.corrmatrix`

I/O, cleaning, denoising, and LRG clustering utilities.

- `load_correlation_matrix(path) -> np.ndarray` -- load `.npy`, `.npz`, `.csv`,
  `.txt`, or `.pkl` files.
- `prepare_correlation_matrix(matrix, zero_diagonal=True, clip=True) -> np.ndarray`
  -- symmetrise, clip to [-1, 1], zero diagonal.
- `marchenko_pastur_denoise(corr, gamma, sigma=1.0) -> np.ndarray` -- eigenvalue
  shrinkage using Marchenko-Pastur support bounds.
- `hierarchical_partitions_from_corr(corr, tau_values, ...) -> list[dict]` -- full
  LRG diffusion clustering pipeline for one matrix.
- `adjusted_rand_index(labels_a, labels_b) -> float` -- partition comparison.
- `compare_partition_sets(set_a, set_b) -> list[dict]` -- cross-matrix ARI.

---

## `multifunbrain.analysis.descriptive`

Descriptive analysis of raw signed dense correlation networks (Section 1).

- `weight_distribution_analysis(corr, n_bins=100) -> dict` -- histogram of
  positive/negative weights, moments, counts, fractions.
- `correlation_spectrum_analysis(corr, gamma=None, sigma=1.0) -> dict` --
  eigenvalue spectrum with optional Marchenko-Pastur comparison (signal/noise
  separation, explained variance).
- `compute_precision_matrix(corr, method="direct", alpha=0.01, gamma=None) -> dict`
  -- precision matrix and partial correlations via pseudoinverse (`"direct"`),
  ORIE denoising (`"orie"`), or GraphicalLasso (`"graphical_lasso"`).
- `signed_laplacian_and_spectrum(corr, normalized=False) -> (L, eigenvalues, eigenvectors)`
  -- compute the signed Laplacian L = |D| - A and its spectrum.
- `signed_laplacian_analysis(corr, normalized=False, n_modes=10) -> dict` --
  full signed Laplacian spectral analysis: frustration index, spectral gap,
  Fiedler vector, leading modes.
- `signed_network_metrics(corr) -> dict` -- dense signed network descriptors:
  positive/negative strength and degree per node, balance ratio, density.
- `descriptive_report(corr, gamma=None, ...) -> dict` -- orchestrates all of the
  above into a single nested dict.

---

## `multifunbrain.analysis.filtering`

Network filtering methods to convert signed matrices into unsigned networks
(Section 2).

- `filter_absolute_threshold(corr, threshold=0.0) -> (nx.Graph, list)` -- take
  absolute values and threshold; extract giant component.
- `filter_split_sign(corr, threshold=0.0) -> dict` -- separate into
  positive-only and negative-only subnetworks.
- `filter_validated(corr, method="disparity", alpha=0.05, ...) -> (nx.Graph, list)`
  -- backbone extraction: `"disparity"` (Serrano 2009), `"lans"`, or
  `"mp_validated"` (reconstruct from signal eigenvalues only).
- `filter_partial_correlation(corr, method="direct", threshold=0.0, ...) -> (nx.Graph, list)`
  -- build network from partial correlations (precision matrix).
- `apply_all_filters(corr, methods=None, threshold=0.0, ...) -> dict` -- apply
  selected filtering methods and return `{name: {"graph": G, "nodes_removed": [...]}}`.

---

## `multifunbrain.analysis.netmetrics`

Standard unsigned network metrics applied after filtering (Section 3).

- `compute_global_metrics(G, weight="weight") -> dict` -- density, clustering,
  shortest path, efficiency, assortativity, components, modularity.
- `compute_node_metrics(G, weight="weight", community_partition=None) -> DataFrame`
  -- degree, strength, clustering, betweenness, closeness, eigenvector centrality;
  with partition: participation coefficient and within-module z-score.
- `detect_communities_louvain(G, weight="weight", resolution=1.0, seed=None) -> dict`
  -- Louvain community detection.
- `detect_communities_spectral(G, n_communities=None, weight="weight") -> dict`
  -- spectral clustering on Laplacian eigengap.
- `degree_distribution_analysis(G, weight=None, n_bins=30) -> dict` -- degree/strength
  distribution with moments.
- `compute_rich_club_curve(G, normalized=True, n_rand=100, seed=None) -> dict`
  -- rich-club coefficient curve.
- `network_summary_report(G, ...) -> dict` -- orchestrates all above into one
  structured output.

---

## `multifunbrain.visualization`

### Pipeline plots

All functions accept an optional `ax=` parameter and return `(fig, ax)`.

- `plot_pipeline_summary(result, figsize=(16, 10)) -> (fig, axes)` -- six-panel
  overview: correlation heatmap, weight distribution, eigenvalue spectrum, signed
  Laplacian, filter comparison, node strength.
- `plot_weight_distribution(result, ax=None, n_bins=80) -> (fig, ax)` --
  positive/negative weight histogram.
- `plot_eigenvalue_spectrum(result, ax=None, show_mp=True) -> (fig, ax)` --
  eigenvalue plot with optional MP bulk shading.
- `plot_signed_laplacian_spectrum(result, ax=None) -> (fig, ax)` -- bar chart of
  signed Laplacian eigenvalues (negative in red, positive in green).
- `plot_correlation_matrix(result, ax=None, cmap="RdBu_r") -> (fig, ax)` --
  diverging-colormap heatmap.
- `plot_filtered_comparison(result, ax=None, metric="density") -> (fig, ax)` --
  bar chart comparing a metric across filtered networks.
- `plot_node_metrics(result, filter_name=None, metrics=(...)) -> (fig, axes)` --
  histograms of per-node metrics.

### LRG visualisation

- `plot_entropy_and_C(ax1, t, Sm1, Csp) -> None` -- dual-axis plot of
  diffusion entropy and clustering coefficient.
- `plot_sankey_matplotlib(partitions, tau_values) -> None` -- Matplotlib Sankey
  diagram of community transitions across scales.
- `plot_sankey(partitions, tau_values) -> None` -- interactive Plotly Sankey.
- `imshow_colorbar_caxdivider(mappable, ax, ...) -> (divider, cax, clb)` --
  colorbar helper.

---

## `multifunbrain.core`

- `hello_brain(name) -> str` -- greeting for sanity checks.
- `band_filter(data, low, high, fs=1.0, order=4) -> np.ndarray` -- Butterworth
  bandpass filter.
- `marchenko_pastur_density(eigenvalues, gamma, sigma=1.0) -> np.ndarray` --
  Marchenko-Pastur density evaluation.

## `multifunbrain.generation`

- `generate_hmn(levels=3, base_module_size=4, p_in=1.0, p_out=0.05, seed=None)`
  -- hierarchical modular network.
- `generate_flower_graph(u=2, v=2, iterations=3)` -- (u, v)-flower graph.
- `generate_brain_timeseries(..., return_time=False)` -- synthetic multichannel
  time series with modular structure.

## `multifunbrain.analysis.lrglib`

- `rho_matrix(tau, L)` -- diffusion kernel normalised to trace 1.
- `entropy(w, steps=600, t1=-2, t2=5)` -- diffusion-based spectral entropy.
- `compute_optimal_threshold(linkage_matrix)` -- partition stability threshold.
- `identify_switching_nodes(partitions, tau_values)` -- track community changes.

## `multifunbrain.analysis.graphutils`

- `get_giant_component(graph, strongly=False)` -- largest connected component.
- `build_correlation_network(timeseries, ...)` -- construct graph from correlations.
- `compute_threshold_stats(graph)` -- threshold sweep statistics.
- `compute_normalized_linkage(distance_matrix)` -- hierarchical clustering linkage.
