"""Live Postgres ground-truth access for the project-brain MCP server.

Read-only by construction: a dedicated read-only role, a read-only
transaction, a statement timeout, a row cap, and this SELECT-only guard.
"""

from __future__ import annotations

import re

# Tokens that must never appear in a read-only query (matched as whole words).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|do|vacuum|reindex|comment|lock|listen|notify|set|reset|"
    r"prepare|execute|begin|commit|rollback|savepoint|refresh)\b",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    """Remove block and line comments so they cannot hide a second statement."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Return (True, "") if sql is a single read-only SELECT/WITH query.

    Returns (False, reason) otherwise. This is a fast-fail UX layer; the
    read-only role and read-only transaction are the real guarantees.
    """
    if not sql or not sql.strip():
        return False, "empty query"

    cleaned = _strip_comments(sql).strip()
    # Drop a single optional trailing semicolon; any remaining ';' means
    # multiple statements.
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if ";" in cleaned:
        return False, "multiple statements are not allowed"

    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "only SELECT / WITH … SELECT queries are allowed"

    match = _FORBIDDEN.search(cleaned)
    if match:
        return False, f"disallowed keyword '{match.group(0).lower()}' (read-only access only)"

    return True, ""
