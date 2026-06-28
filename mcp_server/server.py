"""Project-brain MCP server: live DB ground truth + project knowledge.

stdio transport. Every tool is read-only. Run with: uv run starbot-mcp
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server import conventions, db, knowledge, migrations, settings

mcp = FastMCP("starbot-brain")


# ── Ground truth (live DB, read-only) ─────────────────────────────────────
@mcp.tool()
def list_tables(schema: str = "public") -> list[dict]:
    """List base tables in a schema with estimated row counts (live DB)."""
    return db.list_tables(schema)


@mcp.tool()
def get_schema(table: str, schema: str = "public") -> dict:
    """Return live columns, constraints, and indexes for a table. Use this
    instead of guessing column names — the live schema is ground truth."""
    return db.get_schema(table, schema)


@mcp.tool()
def run_select(sql: str, limit: int = 100) -> dict:
    """Run a read-only SELECT (SELECT/WITH only) and return up to `limit` rows.
    Writes, DDL, and multiple statements are rejected."""
    return db.run_select(sql, limit)


@mcp.tool()
def list_migrations() -> dict:
    """List StarBot SQL migrations and the next migration number to use."""
    return migrations.list_migrations()


# ── Knowledge / conventions (files) ───────────────────────────────────────
@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> dict:
    """Search the compiled knowledge base (gotchas, patterns). Returns ranked
    article bodies + the index for you to synthesize. Check this before acting
    on anything subtle (permissions, migrations, timezones, PostgREST)."""
    return knowledge.search_knowledge(query, limit)


@mcp.tool()
def get_knowledge_index() -> str:
    """Return the knowledge base index (one line per article)."""
    return knowledge.get_knowledge_index()


@mcp.tool()
def get_mistakes(topic: str = "") -> str:
    """Return the bot 'don'ts' log, optionally filtered to a topic keyword."""
    return conventions.get_mistakes(topic or None)


@mcp.tool()
def get_conventions(repo: str = "") -> str:
    """Return workspace coding conventions and rules (AGENTS.md)."""
    return conventions.get_conventions(repo or None)


def main() -> None:
    """Entrypoint. Fail fast if no read-only DB URL is configured."""
    settings.get_database_url()  # raises RuntimeError with guidance if unset
    mcp.run()


if __name__ == "__main__":
    main()
