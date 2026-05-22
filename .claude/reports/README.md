# `.claude/reports/` — Dated analytical reports

Each report captures the findings of a specific analysis: what data
was analysed, with what config, and what was learned.

## Filename convention

```
YYYY-MM-DD_<key-terms-kebab-case>.md
```

Examples: `2026-05-20_repo-inventory.md`,
`2026-05-22_april-global-lrg.md`,
`2026-06-01_co2-vs-rest-multiscale.md`.

## Suggested structure

1. **Context** — what data, what config, what question.
2. **Method** — pointers to scripts/notebooks used.
3. **Findings** — concise numerical results.
4. **Next steps** — open questions / follow-ups.

Reports are **append-only**: once dated, treat as a historical record.
If a later analysis supersedes a finding, write a new report that
references the old one.

For *code/architecture* evolution (refactors, conventions), use
`../history/` instead. For *durable how-tos*, use `../guide/`.
