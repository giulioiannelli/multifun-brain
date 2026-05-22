# Pipeline usage

The three-section pipeline takes a raw signed correlation matrix and
produces a structured `PipelineResult` with descriptive stats, filtered
unsigned networks, and LRG multiscale partitions per filtered network.

## Minimum example

```python
from multifunbrain import PipelineConfig, run_pipeline

cfg = PipelineConfig(gamma=0.19)         # p/n for RMT — set per dataset
result = run_pipeline("path/to/corr.pkl", config=cfg, label="subject01")

result.descriptive["spectrum"]["largest_eigenvalue"]
result.filtered_networks["absolute"]["graph"]
result.lrg_results["absolute"]            # list of partition dicts (one per tau)
```

## Batch and directory runs

```python
from multifunbrain import run_pipeline_batch, run_pipeline_directory, load_results
import pandas as pd

results = run_pipeline_directory("data/correlation_matrices/", config=cfg)
df = pd.concat([r.summary_table() for r in results], ignore_index=True)
```

For a metadata-aware loader (the April batch), see the
[April scaffold scripts](../reports/2026-05-20_repo-inventory.md).

## What each section does

- **Section 1 — descriptive analysis** (`multifunbrain.analysis.descriptive`)
  - Weight distribution (signed network)
  - Eigenvalue spectrum (with MP overlay if `gamma` is set)
  - Precision matrix / partial correlations (`direct` / `orie` / `graphical_lasso`)
  - Signed Laplacian + frustration index
  - Dense signed-network metrics (balance ratio, strength, ...)
- **Section 2 — filtering** (`multifunbrain.processing`)
  - Absolute threshold (`|corr| ≥ th`, auto-percolation if `th=None`)
  - Positive / negative split
  - Backbone extraction (disparity / LANS / MP-validated)
  - Partial-correlation network
- **Section 3 — standard metrics + LRG** (`multifunbrain.analysis.network`,
  `multifunbrain.analysis.lrg`)
  - Global + node metrics
  - Louvain community detection (+ modularity)
  - Optional rich-club curve
  - LRG hierarchical partitions across `tau` scales

## `PipelineConfig` knobs

| Field | Purpose | Default |
|---|---|---|
| `gamma` | Aspect ratio `p/n` for RMT and MP-validated filter | `None` (disables MP) |
| `sigma` | MP noise variance | `1.0` |
| `precision_method` | `direct` / `orie` / `graphical_lasso` | `direct` |
| `precision_alpha` | Regularisation for graphical-lasso | `0.01` |
| `n_signed_modes` | Signed-Laplacian eigenmodes to keep | `10` |
| `filter_methods` | Which filters to apply | `["absolute", "positive", "negative"]` |
| `filter_threshold` | Edge threshold (`None` ⇒ percolation) | `None` |
| `filter_alpha` | Backbone significance level | `0.05` |
| `tau_values` | LRG diffusion scales | `logspace(-2, 1, 6)` |
| `normalized_laplacian` | Use normalised Laplacian in LRG | `True` |
| `run_lrg` / `run_standard_metrics` / `run_community_detection` / `run_rich_club` | Section toggles | `True` / `True` / `True` / `False` |
| `seed` | RNG seed for stochastic algorithms | `None` |

## `PipelineResult` schema (what you get back)

- `result.config` — the `PipelineConfig` used.
- `result.label` — caller-provided label (becomes the row key in summaries).
- `result.n_regions_original`, `result.dropped_regions` — bookkeeping for
  dead-region drops.
- `result.corr_raw`, `result.corr_prepared` — input and cleaned matrices.
- `result.descriptive` — Section 1 output dict.
- `result.filtered_networks[name]` → `{"graph": nx.Graph, "nodes_removed": [...], "percolation": {...}?}`.
- `result.network_analyses[name]` → output of `network_summary_report`.
- `result.lrg_results[name]` → list of partition dicts (one per `tau`).
- `result.summary_table()` → one-row-per-filter DataFrame for concatenation.
- `result.to_dict()` → JSON-serialisable form (numpy/DataFrames flattened).

## `ResultsCollection`

```python
results = load_results("data/correlation_matrices_results/april/global/")
results.labels                    # list of labels
results["band/co2_bpfBOLD/s4"]    # by label
results.filter("co2")             # sub-collection
results.summary_table()           # all rows concatenated
```
