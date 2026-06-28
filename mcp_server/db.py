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


def _strip_string_literals(sql: str) -> str:
    """Replace string-literal CONTENTS with empty placeholders so reserved
    words or semicolons inside literals don't trip the guard. Handles
    single-quoted strings (with '' escapes) and $tag$ dollar-quoted strings.
    Used for ANALYSIS only; the original SQL is what actually executes."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("''")
        elif c == "$":
            m = re.match(r"\$([A-Za-z_]\w*)?\$", sql[i:])
            if m:
                tag = m.group(0)
                end = sql.find(tag, i + len(tag))
                out.append("''")
                if end == -1:
                    break
                i = end + len(tag)
                continue
            out.append(c)
            i += 1
        else:
            out.append(c)
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
    cleaned = _strip_string_literals(cleaned)
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


def run_select(sql: str, limit: int = settings.DEFAULT_ROWS) -> dict:
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


def list_tables(schema: str = "public") -> list[dict]:
    """List base tables in a schema with estimated row counts."""
    query = """
        SELECT c.relname AS table, c.reltuples::bigint AS estimated_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relkind = 'r'
        ORDER BY c.relname
    """
    try:
        with _connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (schema,))
                rows = cur.fetchall()
            conn.rollback()
    except psycopg.Error as exc:
        return [{"error": f"query failed: {str(exc).splitlines()[0]}"}]
    return rows


def get_schema(table: str, schema: str = "public") -> dict:
    """Return columns, constraints, and indexes for a table (live ground truth)."""
    cols_q = """
        SELECT column_name AS column, data_type AS type,
               (is_nullable = 'YES') AS nullable, column_default AS default,
               (is_identity = 'YES') AS identity, ordinal_position AS position
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    cons_q = """
        SELECT conname AS name, pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid = %s::regclass
        ORDER BY conname
    """
    idx_q = """
        SELECT indexname AS name, indexdef AS definition
        FROM pg_indexes
        WHERE schemaname = %s AND tablename = %s
        ORDER BY indexname
    """
    qualified = f"{schema}.{table}"
    try:
        with _connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(cols_q, (schema, table))
                columns = cur.fetchall()
                if not columns:
                    conn.rollback()
                    return {"table": table, "schema": schema, "columns": [],
                            "error": f"table '{qualified}' not found or no columns"}
                try:
                    cur.execute(cons_q, (qualified,))
                    constraints = cur.fetchall()
                except psycopg.Error:
                    conn.rollback()
                    constraints = []
                cur.execute(idx_q, (schema, table))
                indexes = cur.fetchall()
            conn.rollback()
    except psycopg.Error as exc:
        return {"error": f"query failed: {str(exc).splitlines()[0]}"}

    return {
        "table": table,
        "schema": schema,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
    }
