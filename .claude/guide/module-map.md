# Module map — "where does X live?"

Single canonical home per function/class. Phase 1 of the cleanup
retired the back-compat shim files under `multifunbrain/analysis/`
(`corrmatrix.py`, `filtering.py`, `netmetrics.py`, `lrglib.py`) and
collapsed `multifunbrain/core.py`. The public ergonomic surface
(`from multifunbrain.analysis import X` for the common symbols) is
preserved by re-exports inside `multifunbrain/analysis/__init__.py`,
but new code should still prefer the canonical paths below.

## By concern

### I/O — `multifunbrain.io`

| Symbol | Canonical home |
|---|---|
| `load_correlation_matrix` | `io/corrmatrix.py` |
| `load_results` | `io/results.py` |
| `ResultsCollection` | `io/results.py` |

### Preprocessing — `multifunbrain.preprocessing`

| Symbol | Canonical home |
|---|---|
| `detect_dead_regions` | `preprocessing/dead_regions.py` |
| `prepare_correlation_matrix` | `preprocessing/prepare.py` |
| `marchenko_pastur_denoise` | `preprocessing/denoising.py` |
| `marchenko_pastur_density` | `preprocessing/denoising.py` (moved from `core.py` in Phase 1) |

### Processing — `multifunbrain.processing`

| Symbol | Canonical home |
|---|---|
| `filter_absolute_threshold` | `processing/filtering.py` |
| `filter_split_sign` | `processing/filtering.py` |
| `filter_partial_correlation` | `processing/filtering.py` |
| `apply_all_filters` | `processing/filtering.py` |
| `filter_validated` | `processing/backbone.py` |
| `compute_precision_matrix` | `processing/partial_correlation.py` |
| `percolation_threshold` | `processing/percolation.py` |
| `band_filter` | `processing/temporal.py` |
| `matrix_to_graph_giant` (internal helper) | `processing/_giant.py` |

### Analysis — `multifunbrain.analysis`

#### Descriptive — `analysis/descriptive/`

| Symbol | Submodule |
|---|---|
| `weight_distribution_analysis` | `weights.py` |
| `correlation_spectrum_analysis` | `spectrum.py` |
| `signed_laplacian_and_spectrum`, `signed_laplacian_analysis`, `signed_network_metrics` | `signed.py` |
| `descriptive_report` | `report.py` |

#### LRG — `analysis/lrg/`

| Symbol | Submodule |
|---|---|
| `graph_laplacian_and_spectrum`, `rho_matrix`, `entropy` | `kernel.py` |
| `symmetrized_inverse_distance` | `distance.py` |
| `hierarchical_partitions_from_corr`, `compute_optimal_threshold`, `identify_switching_nodes`, `get_moved_nodes`, `get_moved_nodes_interval` | `partitions.py` |

#### Network metrics — `analysis/network/`

| Symbol | Submodule |
|---|---|
| `compute_global_metrics` | `global_metrics.py` |
| `compute_node_metrics` | `node_metrics.py` |
| `detect_communities_louvain`, `detect_communities_spectral` | `community.py` |
| `degree_distribution_analysis`, `compute_rich_club_curve` | `distribution.py` |
| `network_summary_report` | `report.py` |

#### Partition comparison

| Symbol | Canonical home |
|---|---|
| `adjusted_rand_index`, `compare_partition_sets` | `analysis/partition.py` |

#### Graph utilities

| Symbol | Canonical home |
|---|---|
| `get_giant_component`, `get_giant_component_leftoff`, `compute_normalized_linkage`, threshold heuristics | `analysis/graphutils.py` |

### Datasets — `multifunbrain.datasets`

| Symbol | Canonical home |
|---|---|
| `AprilEntry`, `discover_april`, `load_entry`, `label_for`, `entries_to_dataframe` | `datasets/april.py` |

### Pipeline — `multifunbrain.pipeline`

| Symbol | Canonical home |
|---|---|
| `PipelineConfig` | `pipeline/config.py` |
| `PipelineResult`, `sanitise` | `pipeline/result.py` |
| `run_pipeline`, `run_pipeline_batch`, `run_pipeline_directory` | `pipeline/runner.py` |
| `discover_matrices`, `label_from_path`, `SUPPORTED_EXTENSIONS` | `pipeline/discovery.py` |

### Visualization — `multifunbrain.visualization.plotlib`

Largely unchanged from previous structure. Pipeline-specific plots live
in `plotlib/pipeline_plots.py` (`plot_correlation_matrix`,
`plot_lrg_entropy`, `plot_lrg_sankey`, etc.).

### Generation — `multifunbrain.generation`

Synthetic-network generators (`generate_hmn`, etc.).

## Ergonomic re-exports

`multifunbrain.analysis.__init__` and the package-level
`multifunbrain.__init__` re-export the most common public symbols from
sibling packages (`io`, `preprocessing`, `processing`) so notebooks can
write::

    from multifunbrain.analysis import detect_dead_regions  # ergonomic
    # equivalent canonical import:
    from multifunbrain.preprocessing.dead_regions import detect_dead_regions

Library code, CLI code, scripts, and tests should use the canonical
import. The ergonomic surface is only there for notebooks and the
``multifunbrain.notebook`` wildcard namespace.

### What changed in Phase 1

Phase 1 of the multi-phase cleanup (branch `phase-1-retire-shims`)
deleted these files:

- `multifunbrain/analysis/corrmatrix.py`
- `multifunbrain/analysis/filtering.py`
- `multifunbrain/analysis/netmetrics.py`
- `multifunbrain/analysis/lrglib.py`
- `multifunbrain/core.py`

Symbols that used to live in those files are now reachable only via
their canonical homes (and the ergonomic re-exports). The
`hello_brain` helper was inlined into the CLI ``hello`` subcommand;
`marchenko_pastur_density` moved from `core.py` to
`preprocessing/denoising.py`; `band_filter` is exposed via
`processing.temporal` (and the top-level `multifunbrain` namespace).
