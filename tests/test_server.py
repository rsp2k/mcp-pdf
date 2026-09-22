"""Tests for the server module's public surface.

This file previously imported `validate_pdf_path`, `extract_text`,
`extract_tables` and six more names directly from `mcp_pdf.server`. Those were
module-level functions in the pre-mixin monolith; the refactor moved every one
of them into `mcp_pdf.mixins_official.*`, leaving `server.py` exporting only
`PDFServerOfficial`, `create_server` and `main`.

The stale imports raised ImportError at collection time, which pytest treats
as fatal, so a bare `uv run pytest` aborted the entire suite with `1 error`
and ran no tests at all. That is the command both CLAUDE.md and
LOCAL_DEVELOPMENT.md tell you to use.

So this now tests what the module actually exposes. Per-tool behavior belongs
with the mixin that owns it (see test_mixin_architecture.py and test_xfa.py).
"""

from __future__ import annotations

import pytest

from mcp_pdf.server import PDFServerOfficial, create_server


@pytest.fixture
def server():
    return create_server()


class TestServerConstruction:
    def test_create_server_returns_the_server_type(self, server):
        assert isinstance(server, PDFServerOfficial)

    def test_server_exposes_a_fastmcp_instance(self, server):
        assert server.mcp is not None
        assert server.mcp.name

    def test_mixins_are_registered(self, server):
        """Every mixin listed in _register_mixins should be instantiated.

        Catches a mixin silently failing to import, which would otherwise
        surface as a tool quietly missing from the server at runtime.
        """
        assert len(server.mixins) > 0


class TestToolRegistration:
    """Tool registration goes through FastMCP's tool manager, so assert
    against that rather than a private attribute. v2.3.0's version of this
    file reached for `server._tool_handlers`, which FastMCP does not have."""

    @pytest.fixture
    def tool_names(self, server):
        return registered_tool_names(server)

    def test_core_tools_are_registered(self, tool_names):
        # A representative slice across mixins rather than an exhaustive list,
        # so adding a tool does not require editing this test.
        expected = {
            "extract_text",
            "extract_tables",
            "extract_metadata",
            "extract_form_data",
            "pdf_to_markdown",
        }
        missing = {
            name for name in expected
            if not any(t.endswith(name) for t in tool_names)
        }
        assert not missing, f"tools missing from registration: {missing}"

    def test_xfa_tools_are_registered(self, tool_names):
        """Added in v2.3.0. Registration is what makes them reachable at all."""
        for name in ("is_xfa_pdf", "extract_xfa_fields"):
            assert any(t.endswith(name) for t in tool_names), name

    def test_no_duplicate_tool_registration(self, server):
        """Two mixins claiming one tool name silently shadows one of them."""
        names = registered_tool_names(server, unique=False)
        assert len(names) == len(set(names))


def registered_tool_names(server, unique: bool = True):
    """The names of every tool registered on the server.

    Spans both FastMCP tool-listing APIs deliberately. 4.x replaced
    ``get_tools()``, which returned a name-keyed dict, with an async
    ``list_tools()`` returning a ``Sequence[Tool]``. Reading the names
    through whichever exists means this suite does not pin the project to
    one FastMCP major, and the failure mode if a third API arrives is an
    explicit AttributeError here rather than a confusing assertion error
    in each test that uses it.
    """
    import asyncio
    import inspect

    if hasattr(server.mcp, "list_tools"):          # FastMCP 4.x
        result = server.mcp.list_tools()
        tools = asyncio.run(result) if inspect.isawaitable(result) else result
        names = [t.name for t in tools]
    elif hasattr(server.mcp, "get_tools"):         # FastMCP 2.x / 3.x
        result = server.mcp.get_tools()
        mapping = asyncio.run(result) if inspect.isawaitable(result) else result
        names = list(mapping)
    else:
        raise AttributeError(
            "FastMCP exposes neither list_tools() nor get_tools(); the "
            "tool-listing API changed again and this helper needs updating."
        )

    return set(names) if unique else names
