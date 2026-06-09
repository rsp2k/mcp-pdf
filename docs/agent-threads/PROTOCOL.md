# Agent Thread Protocol

Async agent-to-agent coordination via immutable, numbered flat files.

- **Immutable**: once written, a message file is never modified.
- **Sequential**: messages numbered `001`, `002`, `003`, ...
- **Self-describing**: each message carries a metadata header (From / To / Date / Re).
- **Human-readable**: plain markdown.

Reply by creating the next numbered file in the same thread directory:

```
{NNN}-{from}-{2-4 word summary}.md
```

Each message starts with:

```markdown
# Message {NNN}

| Field | Value |
|-------|-------|
| From | {agent} |
| To | {agent} |
| Date | {ISO date} |
| Re | {subject} |

---

{body}
```

See `~/.claude/CLAUDE.md` for the full protocol. Current thread:
`xfa-form-support/` — request to support dynamic Adobe XFA forms.
