# Local Development Guide

This guide is for working **on** mcp-pdf — testing changes against a local Claude Code session before publishing to PyPI. End-user install instructions live in [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Claude Code CLI (or Claude Desktop app)
- Git

System binaries needed at runtime are listed in [README.md → System Dependencies](README.md#system-dependencies). For development you can skip the ones you're not actively touching (e.g. skip `pandoc` if you're not changing `markdown_to_pdf`).

## Setup

```bash
git clone https://github.com/rsp2k/mcp-pdf.git
cd mcp-pdf

# Install dev deps + all optional extras so every tool is exercised
uv sync --all-extras --dev

# Smoke test — does the package even import?
uv run python -c "from mcp_pdf.server import create_server; print('ok')"
```

## Wiring up a local checkout to Claude Code

Three useful patterns, in order of how often you'll want each:

### Pattern 1 — Local source, hot-reload edits

Best when iterating on tool code. `uv run` resolves to the local checkout, so saves are picked up next time the MCP server restarts.

```bash
# From inside the mcp-pdf checkout
claude mcp add -s project pdf-tools-dev -- uv --directory . run mcp-pdf

# Or from anywhere, with an absolute path
claude mcp add -s project pdf-tools-dev -- uv --directory /path/to/mcp-pdf run mcp-pdf
```

The `--` separator is required so the Claude CLI doesn't try to parse `--directory` as one of its own flags.

### Pattern 2 — Specific PyPI version (regression testing)

Useful for confirming that a regression report on a published version actually reproduces.

```bash
claude mcp add -s project pdf-tools-pinned -- uvx --from "mcp-pdf==2026.09.21" mcp-pdf
```

### Pattern 3 — Latest from PyPI (post-publish smoke test)

After running `uv publish`, verify the new version actually works in Claude before announcing it:

```bash
claude mcp add -s project pdf-tools-released -- uvx --refresh --from "mcp-pdf[markdown]" mcp-pdf
```

`--refresh` forces uvx to bypass its cache, otherwise it'll keep using the previous release.

## Running the server outside Claude (for debugging)

Sometimes the MCP transport hides the real failure. Run the server directly:

```bash
uv run mcp-pdf
# Or, equivalently:
uv run python -m mcp_pdf.server
```

Then use a test harness like the FastMCP in-process client, or hit the stdio transport directly with JSON-RPC messages. The server logs go to stderr.

## Tests

```bash
# Full suite
uv run pytest

# A single area
uv run pytest tests/test_server.py

# With coverage
uv run pytest --cov=mcp_pdf

# Security-focused tests live at the repo root (not under tests/)
uv run python test_security_features.py
uv run python test_integration.py
```

## Manual verification before a release

A short list to walk through against a real PDF before publishing. Update version in `pyproject.toml` first.

- [ ] `extract_text` on a small PDF returns text
- [ ] `extract_text` on a 1000+ page PDF chunks correctly (no MCP "response too large" errors)
- [ ] `extract_tables` finds tables in a financial-report-style PDF
- [ ] `pdf_to_markdown` writes to disk by default and the inline escape hatch (`inline=True`) returns full markdown
- [ ] `markdown_to_pdf` converts a real `.md` → PDF via the auto-detected engine (run with `pdf_engine=None` and check the `engine_used` field in the response)
- [ ] `extract_form_data` + `fill_form_pdf` round-trip on an interactive form
- [ ] `add_sticky_notes`, `add_highlights`, `add_stamps` all produce viewable annotations in Acrobat
- [ ] Path traversal blocked: passing `../../etc/passwd` returns a sanitized error
- [ ] Page count limit blocked: a >1000-page PDF returns a sanitized error
- [ ] Error messages don't leak `/home/$USER/` paths (the `sanitize_error_message` helper handles this)

## Publishing pipeline

```bash
# 1. Bump version in pyproject.toml. CalVer, YYYY.MM.DD:
#      version = "2026.09.21"
#    Same-day fix? Append a PEP 440 post-release segment: "2026.09.21.1"
#    Do NOT hardcode it anywhere else; mcp_pdf._version reads package metadata
#    and __init__/server_info both source from there.
#    Add a CHANGELOG.md entry in the same commit.

# 2. Sync uv.lock — easy to forget; the lockfile and pyproject.toml versions
#    must agree or the published wheel and sdist disagree
uv lock

# 3. Clean dist/ — uv publish uploads everything in dist/, so stale files
#    from previous releases cause duplicate-upload errors
rm -rf dist/

# 4. Build
uv build

# 5. Pre-publish PII audit — see ~/.claude/rules/python.md for the
#    full grep recipe. The unpacked-sdist check is non-negotiable.
mkdir -p /tmp/sdist-audit && tar -xzf dist/*.tar.gz -C /tmp/sdist-audit
grep -rnEi 'real-domain|10\.[0-9]+\.|192\.168\.|/home/' /tmp/sdist-audit/

# 6. Validate metadata
uv run twine check dist/*

# 7. Upload (uv publish doesn't read ~/.pypirc — twine does)
uv run twine upload dist/*

# 8. Verify the version is live
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
    https://pypi.org/project/mcp-pdf/$(grep ^version pyproject.toml | cut -d'"' -f2)/
```

Once published, push tags + commits to both remotes:

```bash
git push new-origin main
git push origin main   # gitea mirror
```

## Common gotchas

### "Module not found" after a `git pull`
Run `uv sync --all-extras` again — new optional dependencies don't auto-install.

### MCP server connects but tools don't appear
The Claude Code session caches the tool list at MCP-server-connect time. After adding a new tool, run `/mcp` and reconnect.

### `markdown_to_pdf` errors with `mktexfmt: Did not find entry for byfmt=xelatex`
The host's TeX install is missing format files. Either run `sudo fmtutil-sys --all` to regenerate them, or sidestep by passing `pdf_engine="weasyprint"` (after `pip install weasyprint`) or `pdf_engine="tectonic"`.

### `uv run pytest` aborts at collection with an ImportError
Fixed in v2.3.1, recorded because the wrong diagnosis was in this file for a while. `tests/test_server.py` imported `validate_pdf_path`, `extract_text` and seven more names from `mcp_pdf.server`. Those were module-level functions before the mixin refactor moved them into `mcp_pdf.mixins_official.*`, leaving `server.py` exporting only `PDFServerOfficial`, `create_server` and `main`. pytest treats a collection ImportError as fatal, so the whole suite aborted and ran zero tests.

The lesson worth keeping: if a test file imports from a module that got refactored, check what the module still exports (`grep -nE '^(def|class) ' src/...`) before assuming the failure is about how the test calls things.

### Calling a `@mcp_tool` method in a test
Call it on a mixin instance and `asyncio.run` it, as `tests/test_xfa.py` does. The decorator returns the coroutine function, so `asyncio.run(mixin.extract_xfa_fields(path))` works; there is no `.fn` attribute to unwrap.

### `uv publish` succeeds but PyPI shows old version
PyPI's package-level JSON cache (`/pypi/<pkg>/json`) lags by a minute or two. The version-specific URL (`/pypi/<pkg>/<ver>/`) updates immediately — use that for verification.
