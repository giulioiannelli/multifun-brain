# `.claude/agents/` — Custom subagents

Currently empty. When a task pattern is best handled by a specialised
subagent (with a constrained tool set or a domain-specific prompt), add
a `<name>.md` here.

Format: frontmatter (name, description, model, tools) + system-prompt
body. Invoke via `Agent(subagent_type="<name>", ...)`.
