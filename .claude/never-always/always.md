# Always

## Always use `from __future__ import annotations`

**Why:** Lets us use modern type syntax (`dict[str, int]`, `X | None`)
on Python 3.9+ without runtime parsing of annotations. Every module in
this repo relies on it.

**How to apply:** First non-docstring line in every new `.py` file.
Type hints can then use lowercase generics and union pipes freely.

---

## Always set `gamma` per dataset for RMT comparisons

**Why:** `gamma = n_regions / n_timepoints` is dataset-specific. A wrong
or missing value silently turns the MP-validated filter and the
spectrum-vs-MP overlay into noise (or skips them).

**How to apply:** Get `n_timepoints` from the data collaborator before
running the pipeline. If unavailable, set `gamma=None` and explicitly
note in `.claude/reports/<date>_*.md` that MP-validated filtering /
spectrum overlay are disabled for that run.

---

## Always cite `file:line` when referencing code

**Why:** Lets the user jump straight to the relevant location instead
of grepping the repo.

**How to apply:** Use the format `multifunbrain/pipeline/runner.py:55`
in all communication. Same in commit messages when referring to
specific changes.

---

## Always re-export from the old location when moving a public symbol

**Why:** External notebooks and downstream scripts import from the old
path. Silent removal breaks them.

**How to apply:** Move the canonical definition to the new home; convert
the old module to a re-export shim with a brief docstring pointing at
the new home. Verify with `pytest test/` (existing tests should still
pass with zero modification) before merging.

---

## Always snapshot the `PipelineConfig` next to the results

**Why:** `results.pkl` already contains `PipelineConfig` inside each
`PipelineResult`, but a sibling `config.json` lets reviewers and future
agents understand the run without unpickling.

**How to apply:** Scripts that write `results.pkl` should also write
`config.json` with `json.dumps(config.__dict__, indent=2, default=str)`
(or equivalent) in the same directory.

---

## Always run `pytest test/` after a refactor

**Why:** The test suite is the contract for back-compat. Any refactor
must keep it green.

**How to apply:** Before declaring a refactor done, run the full suite
and confirm zero regressions. Add new tests for new functions in the
same PR.

---

## Always include a date in `.claude/history/` and `.claude/reports/` filenames

**Why:** Both folders are chronological. Date-prefixed names sort
naturally and make it obvious which entry came first.

**How to apply:** Use the format `YYYY-MM-DD_<key-terms-kebab-case>.md`
(e.g. `2026-05-20_repo-inventory.md`). Use today's actual date, not a
session-relative one.

---

## Always treat collaborator pickles as untrusted

**Why:** Pickle is RCE-as-a-feature. Reading collaborator files happens
through the analysis pipeline, never via ad-hoc sandbox
deserialisation.

**How to apply:** When you need to know a pickle's structure, read the
code that produced/consumes it. When inspecting a new batch, run the
metadata-aware loader inside the pipeline rather than scripting `pickle.load`
inline.

---

## Always update `.claude/history/` after a structural change

**Why:** Structural changes (refactors, conventions, new tooling) have
context that doesn't show up in the diff. Without a history entry,
future agents are stuck reverse-engineering intent.

**How to apply:** When you finish a refactor, add
`history/YYYY-MM-DD_<topic>.md` with: what changed, why, what
back-compat measures exist, what's planned next.
