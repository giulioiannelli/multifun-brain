# `.claude/history/` — Evolution log

Append-only chronicle of how the repo got to its current state. Each
file is a dated entry capturing a single significant change — a
refactor, a convention shift, a new tooling decision, an architectural
pivot.

## Filename convention

```
YYYY-MM-DD_<key-terms-kebab-case>.md
```

Examples: `2026-05-20_initial-state.md`,
`2026-05-20_package-reorganization.md`.

## Rules

- **Never edit a history entry after creation.** If a later change
  supersedes it, write a *new* entry that links back to the old one.
- One entry per significant change — don't bundle.
- Keep entries concise but specific: what changed, why, what
  back-compat measures exist, what's still open.
- Use absolute dates only (no "yesterday", "last week").

## What does NOT go here

- Analytical results / numerical findings → `.claude/reports/` instead.
- Durable how-tos → `.claude/guide/` instead.
- Inviolable rules → `.claude/never-always/` instead.
- Routine commits / bug-fixes → `git log` is enough.

## Index (oldest first)

- [`2026-05-20_initial-state.md`](2026-05-20_initial-state.md) — Snapshot
  of the v0.2.0 architecture before the package reorganization.
- [`2026-05-20_package-reorganization.md`](2026-05-20_package-reorganization.md) —
  Split flat `analysis/*.py` modules into scope-based subpackages
  (`io`, `preprocessing`, `processing`, `analysis/{descriptive,lrg,network}`,
  `pipeline`, `datasets`).
- [`2026-05-20_april-scaffold-init.md`](2026-05-20_april-scaffold-init.md) —
  Initial scaffold for the April 2026 correlation-matrix batch
  (metadata-aware loader, orchestration scripts, presentation notebooks).
