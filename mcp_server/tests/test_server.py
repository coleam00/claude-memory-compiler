import asyncio

from mcp_server import server


def test_all_tools_registered():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "list_tables",
        "get_schema",
        "run_select",
        "list_migrations",
        "search_knowledge",
        "get_knowledge_index",
        "get_mistakes",
        "get_conventions",
    }


def test_tool_functions_delegate():
    # Knowledge tools work without a DB; confirm wiring delegates correctly.
    out = server.search_knowledge("postgrest or filter", limit=3)
    assert "matches" in out
