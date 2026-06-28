"""Live Postgres ground-truth access for the project-brain MCP server.

Read-only by construction: a dedicated read-only role, a read-only
transaction, a statement timeout, a row cap, and this SELECT-only guard.
"""

from __future__ import annotations

import re

import psycopg
from psycopg.rows import dict_row

from mcp_server import settings

# Tokens that must never appear in a read-only query (matched as whole words).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|do|vacuum|reindex|comment|lock|listen|notify|set|reset|"
    r"prepare|execute|begin|commit|rollback|savepoint|refresh)\b",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    """Remove line and (possibly nested) block comments so they cannot hide a
    second statement. A scanner is used because regex cannot correctly handle
    nested /* ... /* ... */ ... */ blocks or comments containing '/' or '*'."""
    out: list[str] = []
    i, n, depth = 0, len(sql), 0
    while i < n:
        two = sql[i:i + 2]
        if depth == 0 and two == "--":
            j = sql.find("\n", i)
            if j == -1:
                break
            out.append("\n")
            i = j + 1
        elif two == "/*":
            depth += 1
            i += 2
        elif two == "*/" and depth > 0:
            depth -= 1
            out.append(" ")
            i += 2
        elif depth > 0:
            i += 1
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


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
    if not (re.match(r"select\b", lowered) or re.match(r"with\b", lowered)):
        return False, "only SELECT / WITH … SELECT queries are allowed"

    match = _FORBIDDEN.search(cleaned)
    if match:
        return False, f"disallowed keyword '{match.group(0).lower()}' (read-only access only)"

    return True, ""


def _connect() -> psycopg.Connection:
    """Open a short-lived read-only connection. Fails fast if no URL is set."""
    conn = psycopg.connect(settings.get_database_url(), connect_timeout=10)
    conn.read_only = True
    return conn


def run_select(sql: str, limit: int = 100) -> dict:
    """Run a guarded, row-capped, read-only SELECT and return rows as dicts."""
    ok, reason = is_safe_select(sql)
    if not ok:
        return {"error": reason}

    cap = max(1, min(int(limit), settings.MAX_ROWS))
    # Fetch one extra row to detect truncation; wrap so any SELECT/CTE is capped.
    wrapped = f"SELECT * FROM (\n{sql.rstrip().rstrip(';')}\n) AS _mcp_sub LIMIT {cap + 1}"

    try:
        with _connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {settings.STATEMENT_TIMEOUT_MS}")
                cur.execute(wrapped)
                fetched = cur.fetchall()
                columns = [d.name for d in cur.description] if cur.description else []
            conn.rollback()
    except psycopg.errors.QueryCanceled:
        return {"error": f"query exceeded {settings.STATEMENT_TIMEOUT_MS} ms; add filters"}
    except psycopg.OperationalError as exc:
        return {"error": f"cannot connect to read-only DB: {exc.__class__.__name__}"}
    except psycopg.Error as exc:
        return {"error": f"query failed: {str(exc).splitlines()[0]}"}

    truncated = len(fetched) > cap
    rows = fetched[:cap]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
