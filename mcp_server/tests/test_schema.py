import os

import pytest

from mcp_server import db

needs_db = pytest.mark.skipif(
    not os.environ.get("MCP_DATABASE_URL"),
    reason="MCP_DATABASE_URL not set; live DB integration test skipped",
)


@needs_db
def test_list_tables_includes_known_table():
    tables = {t["table"] for t in db.list_tables()}
    # ai_credits exists in the live schema (see db_snapshot2.json).
    assert "ai_credits" in tables


@needs_db
def test_get_schema_returns_columns():
    out = db.get_schema("ai_credits")
    names = {c["column"] for c in out["columns"]}
    assert {"id", "user_id", "guild_id", "amount", "month_year"} <= names
    # Each column carries type + nullability.
    sample = out["columns"][0]
    assert {"column", "type", "nullable", "default", "position"} <= set(sample)


@needs_db
def test_get_schema_unknown_table():
    out = db.get_schema("definitely_not_a_table_xyz")
    assert out["columns"] == [] or "error" in out
