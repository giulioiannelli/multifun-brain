# `.claude/` — Agent operating manual

Start here when you (or an agent) need to act in this repo.

## Quick map

| Folder | What lives there | When to read |
|---|---|---|
| `guide/` | Durable how-tos (best practices, pipeline usage, results tracking, data layout, module map) | Before doing any non-trivial task |
| `never-always/` | Inviolable rules (`never.md`, `always.md`) | Before destructive actions, before writing new code |
| `history/` | Append-only chronicle of code/architecture/practice changes (dated entries) | When you need to understand *why* something is the way it is |
| `reports/` | Dated analytical reports (results writeups, numerical findings) | When you want to know what previous analyses concluded |
| `skills/` | Custom Skills (currently empty; populated as patterns emerge) | When the user asks about a custom skill or workflow |
| `commands/` | Custom slash-commands (currently empty) | When you see `/<name>` and want to know if it's local |
| `hooks/` | Pre/post hooks (currently empty) | When a hook fires and you need its definition |
| `agents/` | Custom subagent prompts (currently empty) | When invoking a non-default subagent |
| `settings.local.json` | Per-machine Claude Code settings (permissions, env) | When debugging permission denials |

## Entry point for any session

The repo root `CLAUDE.md` is loaded automatically. It points back here for details. Treat that file as the front door, this folder as the rest of the house.

## Updating this folder

- Updating a durable how-to → edit `guide/<topic>.md` in place.
- Logging a code/architecture change → add a new `history/YYYY-MM-DD_<topic>.md` (never edit history entries after creation).
- Recording an analytical result → add `reports/YYYY-MM-DD_<topic>.md`.
- Adding an inviolable rule → append to `never-always/never.md` or `always.md` (additive only).

See `guide/best-practices.md` for the broader contribution norms.
