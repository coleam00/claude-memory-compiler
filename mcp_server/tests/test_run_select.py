import os

import pytest

from mcp_server import db

needs_db = pytest.mark.skipif(
    not os.environ.get("MCP_DATABASE_URL"),
    reason="MCP_DATABASE_URL not set; live DB integration test skipped",
)


def test_guard_rejection_returns_error_dict_without_db():
    # No DB needed: a disallowed query must be rejected before connecting.
    out = db.run_select("DELETE FROM users")
    assert "error" in out
    assert "read-only" in out["error"].lower() or "select" in out["error"].lower()


@needs_db
def test_select_returns_rows_and_caps():
    out = db.run_select("SELECT generate_series(1, 500) AS n", limit=100)
    assert out["columns"] == ["n"]
    assert out["row_count"] == 100
    assert out["truncated"] is True
    assert out["rows"][0] == {"n": 1}


@needs_db
def test_limit_hard_capped():
    out = db.run_select("SELECT generate_series(1, 5000) AS n", limit=99999)
    assert out["row_count"] <= db.settings.MAX_ROWS
