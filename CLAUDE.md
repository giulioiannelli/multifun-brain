# multifun-brain — Project memory

Brain-network analysis toolkit for **signed fMRI correlation matrices**.
The pipeline turns those signed matrices into clean **non-negative
networks** (via filtering / backbone extraction), then runs the
**Laplacian Renormalisation Group (LRG)** to expose connectivity
structure across multiple diffusion scales. The headline use-case is
comparing **hypercapnia (CO2) vs resting-state** fMRI contrasts at
multiple processing variants and frequency bands.

**Results are presented in the dashboard** (`dashboard/`, FastAPI +
React). Collaborators and the PI look at results there, not in notebooks.

## Quick map

| Where | Purpose |
|---|---|
| `multifunbrain/io/` | File I/O: `load_correlation_matrix`, `load_results`, `ResultsCollection` |
| `multifunbrain/preprocessing/` | Dead-region detection, matrix prep, MP denoising |
| `multifunbrain/processing/` | Signed→unsigned filtering, backbone, partial corr, percolation, temporal |
| `multifunbrain/analysis/descriptive/` | Weights / spectrum / signed-Laplacian / report (Section 1) |
| `multifunbrain/analysis/lrg/` | Diffusion kernel, distance, hierarchical partitions (Section 3 — LRG) |
| `multifunbrain/analysis/network/` | Global / node / community / distribution / report (Section 3 — standard) |
| `multifunbrain/analysis/partition.py` | ARI, partition-set comparison |
| `multifunbrain/datasets/` | Dataset-specific loaders (April batch lives here) |
| `multifunbrain/pipeline/` | `PipelineConfig`, `PipelineResult`, runners, discovery |
| `multifunbrain/visualization/plotlib/` | All plot functions |
| `multifunbrain/generation/` | Synthetic-network generators |
| `dashboard/` | **Presentation layer**: FastAPI backend (`backend/serializers/` → JSON specs) + React/Plotly/Cytoscape frontend |
| `quickstart.py`, `SETUP.md` | One-command collaborator setup: detect raw data → ingest → build → serve |
| `scripts/raw_ingest/` | Raw timecourses → EMD bands → correlation → pipeline bundles |
| `scripts/april/` | Thin orchestration for the April batch (global → bands → patients) |
| `notebooks/` | Personal scratch exploration only (`notebooks/april/` is legacy, stale data paths) |
| `test/` | 62+ pytest tests |
| `docs/` | API reference, usage, development guides (MkDocs) |
| `data/` | gitignored; collaborator correlation matrices + atlases |

### Agent-facing detail in `.claude/`

- **`.claude/CLAUDE.md`** — this file's full sibling; treat the two as
  the same document (root one is canonical, auto-loaded).
- **`.claude/guide/`** — durable how-tos. Read before any non-trivial task.
  - `dashboard.md` (**read before touching `dashboard/`**), `best-practices.md`,
    `pipeline-usage.md`, `results-tracking.md`, `data-layout.md`,
    `module-map.md`, `plotting.md`.
- **`.claude/never-always/`** — inviolable rules (`never.md`, `always.md`).
- **`.claude/history/`** — append-only chronicle of architecture changes.
- **`.claude/reports/`** — dated analytical reports.
- **`.claude/{skills,commands,hooks,agents}/`** — placeholders, populate as patterns emerge.

## Key invariants

1. **Signed networks need special treatment** — signed Laplacian
   `L = |D| - A`, frustration index, balance ratio. Standard unsigned
   network methods don't apply directly.
2. **NaN ≠ zero correlation.** NaN rows/columns indicate dead brain
   regions; they must be **dropped**, never zero-filled
   (`detect_dead_regions` + `prepare_correlation_matrix` handle this).
3. **`gamma = n_regions / n_timepoints` is dataset-specific** for the
   Marchenko–Pastur edge. Get it from the collaborator; fall back to
   `gamma=None` (no MP overlay) only when unavailable, and record the
   decision in `.claude/reports/`.
4. **Pickle is RCE.** Read collaborator pickles via
   `multifunbrain.io.corrmatrix.load_correlation_matrix` (the
   intentional, documented call site), never via ad-hoc sandbox
   `pickle.load`.
5. **Single canonical home per function.** No code duplication across
   files — legacy import paths are re-export shims (see
   `.claude/guide/module-map.md`). Removing the shims is a future PR.

## Workflow norms

- New analysis → script in `scripts/<topic>/` (thin, argparse, ≤80 lines).
- New presentation → **dashboard panel** (serializer in
  `dashboard/backend/serializers/` + builder in
  `dashboard/frontend/src/components/plots/`). Reusable computation goes
  into `multifunbrain/` first. Recipes: `.claude/guide/dashboard.md`.
- New library function → goes in the scope-based subpackage that fits.
  See `.claude/guide/module-map.md`.
- Tests for new code in `test/test_<topic>.py`.
- Run `pytest test/` + `ruff check multifunbrain/ dashboard/backend/`
  before declaring done; for frontend changes also `npx tsc --noEmit` +
  `npm run build` in `dashboard/frontend/`, then look at it in the running
  app. CI (`.github/workflows/ci.yml`) runs pytest + ruff on 3.10–3.12.

## Active context

- **Branching**: `main` is the integration branch; feature work on
  short-lived branches → PR → merge when CI is green.
- **Focus (2026-11)**: make the dashboard the complete presentation tool.
  The Network tab (2-D graph, metrics, communities) is the weakest part
  and the next target.
- **Datasets**: result bundles under `data/correlation_matrices_results/`
  — `schaefer100_april2026/`, `schaefer100_november2025/`,
  `harvardoxford48/` — each with `global/`, `bands/`, `patients/`.
  Raw timecourses in `data/raw_data_<atlas><batch>/`. April loader:
  `multifunbrain.datasets.april`.
- **Analysis order** (per Daniele / Giulio collaboration):
  global → bands → patients.

## When to update what

| Event | Where to write |
|---|---|
| Analysis result / numerical finding | `.claude/reports/YYYY-MM-DD_<topic>.md` |
| Refactor / architecture change | `.claude/history/YYYY-MM-DD_<topic>.md` |
| New inviolable rule | append to `.claude/never-always/{never,always}.md` |
| New how-to / pattern | edit `.claude/guide/<topic>.md` in place |
| Recent project context | this file (`CLAUDE.md`), keep it concise |
| Conventional commits | `git log` (no extra file needed) |
