# 2026-05-20 — Repo inventory + April-batch discovery

## Context

Start of the `cleaning-repository` session. The user asked for (1) an
inventory of the repo state, (2) an agent-facing infrastructure scaffold,
and (3) a plan to start analysing the new April 2026 correlation-matrix
batch from Daniele.

## Method

Three parallel `Explore` subagents:

1. Inventory `data/correlation_mat_april_data/` (file count, naming,
   stratification dimensions).
2. Survey `multifunbrain/` package modules + tests + notebooks.
3. Audit `.claude/`, `AGENTS.md`, `CLAUDE.md`, `docs/`, `notebooks/` for
   existing agent-facing guidance.

Plan written to `/home/opisthofulax/.claude/plans/we-need-to-make-luminous-scone.md`.

## Findings — package state

- `multifunbrain` v0.2.0, three-section pipeline implemented
  (`descriptive` → `filtering` → `LRG multiscale`), 62 tests passing.
- LRG kernel is native (no external dep): `analysis/lrglib.py` defines
  `rho_matrix`, `entropy`, hierarchical partition machinery.
- Module boundaries blurred — `analysis/corrmatrix.py` mixes I/O,
  preprocessing, and LRG; `descriptive.py` lumps weights/spectrum/
  precision/signed-Laplacian together; `pipeline.py` is one big file.
  → Reorg done in this session (see `history/2026-05-20_package-reorganization.md`).
- Pre-existing lint: 82 ruff errors (mostly F401, UP035, I001 in legacy
  modules); deprecated `pyproject.toml` ruff config keys. Out of
  today's scope.

## Findings — April batch (`data/correlation_mat_april_data/`)

- **169 MB**, **226 correlation-matrix pickles** + 10 histogram PNGs + 1
  consolidated `All_Emd_diz.pkl` (150 MB, deferred).
- Stratified by:
  - **Contrast** (2): `co2`, `rest`.
  - **Processing variant** (5): `bpfBOLD`, `bpfVASO`, `MIRNoise_bold`,
    `optcom_bold`, `optcomMIRDenoised_bold`.
  - **Band** (3): `s4`, `s5`, `sstar` (IMF slow modes).
  - **Subject** (6): `sub-00246757`, `sub-00259685`, `sub-00307729`,
    `sub-00308305`, `sub-VA11266`, `sub-VA9757`.
- Three aggregation levels:
  - `freq-contrast-global/` — 10 GLOBAL files (across subjects + bands).
  - `freq-contrast-inter/` — 30 per-band aggregates.
  - `freq-user-constrast-inter/` — 180 per-subject × band files.
- No existing notebook references this directory yet → fresh from
  collaborator, no prior analysis in tracked code.

## Findings — agent infrastructure

- `.claude/` had only `settings.local.json` (minimal permissions).
- `AGENTS.md` existed (16 lines, generic). No `CLAUDE.md` at root.
- No structured index of where code lives or what rules apply.

## Decisions

- **Branch strategy:** committed the uncommitted v0.2.0 pipeline
  refactor on `main` first (1 cohesive commit), then branched
  `april-scaffold` for today's work.
- **Reorganization scope:** done in one session; back-compat shims keep
  legacy import paths working. Removing the shims is a future PR.
- **Agent infra layout:** `.claude/{guide,never-always,history,reports,
  skills,commands,hooks,agents}` + root `CLAUDE.md`.
- **April analysis order:** global → bands → patients (smoke-test
  staging).
- **Gamma:** unknown for April matrices; default to `gamma=None`
  pending Daniele's `n_timepoints` numbers; record decisions in a
  dated April-results report when we have output.

## Next steps

- Run `scripts/april/00_inventory.py` to validate the
  metadata-aware loader against the real directory.
- Run `scripts/april/01_run_global.py` and check whether the existing
  `load_correlation_matrix` handles Daniele's pickle format unchanged.
  If it returns a dict (not a bare array), add a thin structural
  adapter in `multifunbrain/datasets/april.py:load_entry`.
- Ask Daniele for `n_timepoints` per acquisition to enable
  MP-validated filtering and the MP overlay.
- After per-file pickles validate, decide whether to also process
  `All_Emd_diz.pkl` (the 150 MB consolidated dataset).
