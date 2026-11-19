# 2026-07-22 — Dashboard extra features (edge styling, LRG network+brain, sparsify propagation, Compare tab)

Branch `worktree-dashboard+extra-feats` (merged the `dashboard` tip `1f8ddc7` in as
the base). Six requested fixes, all implemented + adversarially reviewed + tested.

## Library foundations (`multifunbrain/analysis/lrg/`)
Promoted reusable analysis out of `scripts/april/` (invariant #5, single home):
- `partitions.py::linkage_at_tau_min(graph)` — LRG diffusion linkage at
  τ_min = 1/λ_max. `_fig_common.py` now re-exports it.
- `compare.py::compare_hierarchies(Z_a, leaf_a, Z_b, leaf_b, null_per_atlas=None)`
  → `HierarchyComparison` (Baker's γ, cophenetic r, per-ROI cophenetic-rank shift,
  calibrated shift vs null, pooled null). `null_shift_distribution(...)` wraps
  `cached_surrogate_linkages` (strength-preserving surrogates, disk-cached). These
  are the matplotlib-free compute core of the old `_handoff_compare.build_fig3`.
- `layout.py` (new) — `diffusion_distance_mds(graph, tau)` → 2-D MDS embedding of
  the LRG diffusion distance (the "LRG-distance-inspired" node layout).

## Feature 1 — edge colour + width laws (Network tab)
`network_spec` emits `w_min/w_max/w_absmax/w_absmin/has_negative`. `NetworkGraph.tsx`
colours edges by weight (diverging RdBu if signed, sequential magma on |w| else)
and scales width by a selectable law: linear / sqrt / log / rank(percentile).
Controls "Edge colour" + "Edge width law" in `ExploreView`.

## Feature 3 — square network graph
`.network-square` CSS (`aspect-ratio:1/1`, centred, max-width 720) wraps Cytoscape
(fills 100%/100%) instead of a fixed 560×full-width rectangle.

## Feature 4 — LANS + MP-validated sparsification
New shared `dashboard/backend/graphs.py::derive_graph(result, fname, sparsify,…)`
(the single sparsify dispatcher, extended with `lans` and `mp_validated`;
mp_validated restores the unit diagonal before eigen-thresholding since
`corr_prepared` = C−I). Both added to the Sparsify dropdown.

## Feature 5 — sparsify propagation → LRG & Brain-3D
Sparsify control lifted to the shared subbar (drives Network/LRG/Brain-3D).
`brain3d.render` takes sparsify (re-derives edges). LRG serializers route through
`_lrg_entries(result, fname, sparsify,…)`: stored `lrg_results` for `filter`, else
`_recompute_entries` builds the τ-grid on the sparsified backbone (cached on the
result object). Leaf naming/colours recovered via `node_ids` (survivor positions),
including the filter case where the giant component dropped nodes (n_leaves≠100).

## Feature 2 — LRG-layout clustered network + partition glass brain (LRG tab)
- `lrg_network_spec` — MDS(LRG-distance) coords + edges at the current τ, nodes in
  linkage-leaf order so the client colours by the SAME cut as the dendrogram
  (`clusterLeaves` in `figures.ts` mirrors `glassbrain.leaf_cluster_ids`). Cut
  dragging recolours instantly; only τ refetches.
- `glassbrain.py::partition_html` + `GET /api/lrg_brain` — survivor centroids
  coloured by community at (τ, cut) via nilearn `view_markers`. Degrades to an
  error page for atlases without Schaefer centroids (HarvardOxford48).

## Feature 6 — Brain-comparison tab
- `serializers/compare.py` + `GET /api/compare` (two A/B selections) →
  {baker γ, cophenetic r, n_common, per-ROI shift/calibrated, observed+pooled_null,
  top reorganised/stable}. Built on a **LANS backbone** (as fig6) via `derive_graph`,
  relabelled to **original atlas indices** so two results with different dead-region
  drops still align ROI-to-ROI and map onto the parcellation.
- `glassbrain.scalar_filled_html` + `GET /api/compare/brain` — the per-ROI
  (calibrated) cophenetic shift painted on the Schaefer parcellation NIfTI (RdBu
  diverging when a null is on, magma otherwise). Mirrors `fig6_compare.pdf`.
- `CompareView.tsx` + "Compare" tab: two faceted `SelectorBar`s (group-average or
  single-patient), null toggle (none | strength-preserving, cached; ~14 s first
  compute on LANS), metrics strip, glass brain, observed-vs-null histogram, top
  reorganised/stable bars.

## Bugs found + fixed during verification / adversarial review
1. **Cut-height rounding** (`glassbrain.render_partition`) — the LRU cache key
   rounded `cut_height` to 6 dp, dropping the flat-threshold merge (the optimal cut
   sits exactly on a merge) → the brain showed one extra community vs the
   dendrogram. Now passed at full precision.
2. **LRG clustered network dropped ~90% of edges** — the `/plot` route's
   `edge_quantile=0.9` default overrode `lrg_network_spec`'s intended `0.0`; LrgView
   now sends `edge_quantile=0`.
3. **Disconnected-graph layout collapse** (`diffusion_distance_matrix`) — infinite
   cross-component distance was mapped to `0.0` (coincident); now to a large finite
   value so MDS separates components (common with aggressive sparsify backbones).
4. **Stale strength-null after re-run** — surrogate disk-cache key now includes the
   `results.pkl` mtime; and `null_shift_distribution` encodes the rng seed in the
   cache filename so a self-comparison still draws two independent surrogate sets.
5. **Slash-in-cache-label** — sanitised dataset/label ids for surrogate filenames.

## Tests / lint
New: `test/test_lrg_layout.py`, `test/test_dashboard_graphs.py`, additions to
`test/test_lrg_compare.py` and `test/test_lrg_dashboard.py` (serializer tests
skip when the gitignored Schaefer atlas file is absent). `pytest` **209 pass**
(3 atlas-gated skips in a bare worktree), ruff clean on all changed paths,
`npm run build` (tsc) green. Verified end-to-end in headless-driven Chrome on
`:8011` against the main checkout's `data/correlation_matrices_results`.
