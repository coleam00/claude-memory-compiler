# Project-Brain MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python stdio MCP server in `claude-memory-compiler/mcp_server/` that gives every agent (Claude Code + Codex) live read-only DB ground truth plus searchable project knowledge/conventions, to reduce repeated agent mistakes.

**Architecture:** A `FastMCP` stdio server exposes eight tools. DB tools hit the live Supabase Postgres through a dedicated read-only role (`MCP_DATABASE_URL`), guarded by a SELECT-only parser, a read-only transaction, a statement timeout, and a row cap. Knowledge/convention tools read the existing `knowledge/` markdown, `.claude/mistakes.md`, and `AGENTS.md` directly (no sub-agent spawning). Pure-logic modules (guard, migrations, knowledge, conventions) are unit-tested; DB introspection is integration-tested and skipped when no DB URL is set.

**Tech Stack:** Python 3.12, `uv`, `mcp` (official MCP Python SDK / FastMCP), `psycopg[binary]` (psycopg 3), `python-dotenv` (already present), `pytest`.

## Global Constraints

- Python `>=3.12`; managed with `uv` (run via `uv run …` from `claude-memory-compiler/`).
- Package lives at `claude-memory-compiler/mcp_server/`; tests at `claude-memory-compiler/mcp_server/tests/`.
- **Read-only only.** No tool may write to the DB or filesystem (except this plan's own source files). No applying migrations.
- **No Anthropic/Claude Agent SDK calls** at server runtime. Do **not** import or reuse `scripts/query.py`'s `run_query`.
- DB access **requires** `MCP_DATABASE_URL`; the server process must **fail fast** at startup with a clear message if it is unset (no fallback to a writable connection). Importing modules must NOT fail — only `db.connect()` / server startup require the URL.
- Credentials are never logged and never returned in any tool result.
- `run_select` is SELECT-only, runs in a read-only transaction with `statement_timeout = 5s`, and caps rows (default 100, hard max 1000).
- Ruff line-length 100 (matches existing `pyproject.toml`).
- Reuse existing readers from `scripts/utils.py` (`read_wiki_index`, `list_wiki_articles`) rather than re-implementing markdown discovery.

---

### Task 1: Package scaffold, settings, dependencies

**Files:**
- Create: `claude-memory-compiler/mcp_server/__init__.py`
- Create: `claude-memory-compiler/mcp_server/settings.py`
- Create: `claude-memory-compiler/mcp_server/.env.example`
- Create: `claude-memory-compiler/mcp_server/tests/__init__.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_settings.py`
- Modify: `claude-memory-compiler/pyproject.toml`
- Modify: `claude-memory-compiler/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `settings.COMPILER_ROOT: Path`, `settings.WORKSPACE_ROOT: Path`, `settings.SCRIPTS_DIR: Path`, `settings.KNOWLEDGE_DIR: Path`, `settings.INDEX_FILE: Path`, `settings.MIGRATIONS_DIR: Path`, `settings.MISTAKES_FILE: Path`, `settings.AGENTS_FILE: Path`
  - `settings.MAX_ROWS: int = 1000`, `settings.DEFAULT_ROWS: int = 100`, `settings.STATEMENT_TIMEOUT_MS: int = 5000`
  - `settings.get_database_url() -> str` — returns `MCP_DATABASE_URL` or raises `RuntimeError` with a clear message.

- [ ] **Step 1: Add dependencies and a console script to `pyproject.toml`**

Edit `claude-memory-compiler/pyproject.toml` so `[project]` dependencies include the new packages and a script is registered:

```toml
[project]
name = "llm-personal-kb"
version = "0.1.0"
description = "Personal knowledge base compiled from AI conversations - inspired by Karpathy's LLM KB architecture"
requires-python = ">=3.12"
dependencies = [
    "claude-agent-sdk>=0.1.29",
    "python-dotenv>=1.0.0",
    "tzdata>=2024.1",
    "mcp>=1.2.0",
    "psycopg[binary]>=3.1",
]

[project.scripts]
starbot-mcp = "mcp_server.server:main"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["mcp_server/tests"]
```

- [ ] **Step 2: Install dependencies**

Run: `cd claude-memory-compiler && uv sync --extra dev`
Expected: resolves and installs `mcp`, `psycopg`, `pytest` with no errors.

- [ ] **Step 3: Write the failing test for settings**

Create `claude-memory-compiler/mcp_server/tests/test_settings.py`:

```python
import importlib

import pytest

from mcp_server import settings


def test_paths_resolve_to_real_locations():
    assert settings.KNOWLEDGE_DIR.name == "knowledge"
    assert settings.SCRIPTS_DIR.name == "scripts"
    # Workspace root is the parent of the compiler repo.
    assert settings.WORKSPACE_ROOT == settings.COMPILER_ROOT.parent
    # Migrations live in the sibling StarBot repo.
    assert settings.MIGRATIONS_DIR.parts[-3:] == ("scripts", "migrations") or \
        settings.MIGRATIONS_DIR.name == "migrations"
    assert settings.AGENTS_FILE.name == "AGENTS.md"
    assert settings.MISTAKES_FILE.name == "mistakes.md"


def test_get_database_url_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MCP_DATABASE_URL", raising=False)
    # Force a clean reload so module-level dotenv loading does not repopulate it.
    importlib.reload(settings)
    monkeypatch.delenv("MCP_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError) as exc:
        settings.get_database_url()
    assert "MCP_DATABASE_URL" in str(exc.value)


def test_get_database_url_returns_when_set(monkeypatch):
    monkeypatch.setenv("MCP_DATABASE_URL", "postgresql://ro@localhost/db")
    assert settings.get_database_url() == "postgresql://ro@localhost/db"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server'` (package not created yet).

- [ ] **Step 5: Create the package files**

Create `claude-memory-compiler/mcp_server/__init__.py`:

```python
"""StarBot project-brain MCP server."""
```

Create `claude-memory-compiler/mcp_server/tests/__init__.py` (empty file).

Create `claude-memory-compiler/mcp_server/settings.py`:

```python
"""Paths and configuration for the project-brain MCP server.

Importing this module never raises. Only ``get_database_url()`` enforces the
presence of a database connection string, so unit tests that need no DB can
import freely.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# mcp_server/ -> compiler root
COMPILER_ROOT = Path(__file__).resolve().parent.parent
# Workspace root holds the StarBot/dashboard/nomads repos + AGENTS.md + .claude/
WORKSPACE_ROOT = COMPILER_ROOT.parent

SCRIPTS_DIR = COMPILER_ROOT / "scripts"
KNOWLEDGE_DIR = COMPILER_ROOT / "knowledge"
INDEX_FILE = KNOWLEDGE_DIR / "index.md"

MIGRATIONS_DIR = WORKSPACE_ROOT / "StarBot" / "src" / "scripts" / "migrations"
MISTAKES_FILE = WORKSPACE_ROOT / ".claude" / "mistakes.md"
AGENTS_FILE = WORKSPACE_ROOT / "AGENTS.md"

# Load the server's own .env (git-ignored) if present.
load_dotenv(COMPILER_ROOT / "mcp_server" / ".env")

# Query safety limits.
DEFAULT_ROWS = 100
MAX_ROWS = 1000
STATEMENT_TIMEOUT_MS = 5000


def get_database_url() -> str:
    """Return the read-only DB connection string or fail fast.

    Raises:
        RuntimeError: if MCP_DATABASE_URL is not set. We never fall back to a
        writable connection.
    """
    url = os.environ.get("MCP_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "MCP_DATABASE_URL is not set. The project-brain MCP server requires a "
            "dedicated READ-ONLY Postgres connection string. Set it in "
            "claude-memory-compiler/mcp_server/.env or the environment. See "
            "docs for the one-time read-only role setup."
        )
    return url
```

Create `claude-memory-compiler/mcp_server/.env.example`:

```bash
# Connection string for the dedicated READ-ONLY Postgres role (mcp_ro).
# Copy to mcp_server/.env and fill in. NEVER commit the real value.
MCP_DATABASE_URL=postgresql://mcp_ro:PASSWORD@HOST:5432/postgres
```

- [ ] **Step 6: Ignore the server `.env`**

Append to `claude-memory-compiler/.gitignore`:

```gitignore
mcp_server/.env
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_settings.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
cd claude-memory-compiler
git add pyproject.toml uv.lock .gitignore mcp_server/__init__.py mcp_server/settings.py mcp_server/.env.example mcp_server/tests/__init__.py mcp_server/tests/test_settings.py
git commit -m "feat(mcp): scaffold project-brain MCP server package and settings"
```

---

### Task 2: SQL safety guard

**Files:**
- Create: `claude-memory-compiler/mcp_server/db.py` (guard portion only this task)
- Create: `claude-memory-compiler/mcp_server/tests/test_sql_guard.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `db.is_safe_select(sql: str) -> tuple[bool, str]` — returns `(True, "")` for a single read-only `SELECT`/`WITH … SELECT`, else `(False, reason)`.

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_sql_guard.py`:

```python
import pytest

from mcp_server.db import is_safe_select

SAFE = [
    "SELECT 1",
    "select id, name from users where guild_id = 123",
    "  SELECT * FROM events ORDER BY created_at DESC  ",
    "WITH recent AS (SELECT * FROM events) SELECT * FROM recent",
    "SELECT count(*) FROM tokens;",  # single trailing semicolon allowed
]

UNSAFE = [
    ("", "empty"),
    ("INSERT INTO users VALUES (1)", "insert"),
    ("UPDATE users SET name = 'x'", "update"),
    ("DELETE FROM users", "delete"),
    ("DROP TABLE users", "drop"),
    ("ALTER TABLE users ADD COLUMN x int", "alter"),
    ("TRUNCATE users", "truncate"),
    ("GRANT SELECT ON users TO bob", "grant"),
    ("SELECT 1; DROP TABLE users", "multiple statements"),
    ("SELECT 1; SELECT 2", "multiple statements"),
    ("SELECT * FROM users; -- comment", "multiple statements"),
    ("WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x", "delete"),
    ("COPY users TO '/tmp/x'", "copy"),
    ("CALL do_thing()", "call"),
    ("EXPLAIN ANALYZE DELETE FROM users", "not a select"),
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    ok, reason = is_safe_select(sql)
    assert ok, reason


@pytest.mark.parametrize("sql,_label", UNSAFE)
def test_unsafe_queries_rejected(sql, _label):
    ok, _ = is_safe_select(sql)
    assert not ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_sql_guard.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_safe_select'`.

- [ ] **Step 3: Write the guard implementation**

Create `claude-memory-compiler/mcp_server/db.py`:

```python
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
    # Allow exactly one optional trailing semicolon.
    cleaned = cleaned.rstrip(";").strip() if cleaned.count(";") == 1 and \
        cleaned.rstrip().endswith(";") else cleaned

    if ";" in cleaned:
        return False, "multiple statements are not allowed"

    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "only SELECT / WITH … SELECT queries are allowed"

    match = _FORBIDDEN.search(cleaned)
    if match:
        return False, f"disallowed keyword '{match.group(0).lower()}' (read-only access only)"

    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_sql_guard.py -v`
Expected: PASS (all parametrized cases green).

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/db.py mcp_server/tests/test_sql_guard.py
git commit -m "feat(mcp): add SELECT-only SQL safety guard"
```

---

### Task 3: Read-only query execution

**Files:**
- Modify: `claude-memory-compiler/mcp_server/db.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_run_select.py`

**Interfaces:**
- Consumes: `is_safe_select` (Task 2), `settings.get_database_url`, `settings.DEFAULT_ROWS`, `settings.MAX_ROWS`, `settings.STATEMENT_TIMEOUT_MS` (Task 1).
- Produces: `db.run_select(sql: str, limit: int = 100) -> dict` returning `{"columns": list[str], "rows": list[dict], "row_count": int, "truncated": bool}` or `{"error": str}`.

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_run_select.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_run_select.py -v`
Expected: FAIL — `AttributeError: module 'mcp_server.db' has no attribute 'run_select'`.

- [ ] **Step 3: Implement connection + run_select**

Add to the top imports of `claude-memory-compiler/mcp_server/db.py`:

```python
import psycopg
from psycopg.rows import dict_row

from mcp_server import settings
```

Append to `claude-memory-compiler/mcp_server/db.py`:

```python
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
```

Note: `conn.read_only = True` makes the whole transaction read-only at the protocol level; `SET LOCAL statement_timeout` bounds runtime; the wrap enforces the row cap regardless of the user's own LIMIT.

- [ ] **Step 4: Run tests to verify they pass (DB tests skip without a URL)**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_run_select.py -v`
Expected: `test_guard_rejection_returns_error_dict_without_db` PASS; the two `@needs_db` tests show SKIPPED unless `MCP_DATABASE_URL` is set. If you have the read-only role configured, run with it set and confirm they PASS.

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/db.py mcp_server/tests/test_run_select.py
git commit -m "feat(mcp): add read-only row-capped run_select"
```

---

### Task 4: Schema introspection

**Files:**
- Modify: `claude-memory-compiler/mcp_server/db.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_schema.py`

**Interfaces:**
- Consumes: `_connect` (Task 3), `settings` (Task 1).
- Produces:
  - `db.list_tables(schema: str = "public") -> list[dict]` → `[{"table": str, "estimated_rows": int}]`
  - `db.get_schema(table: str, schema: str = "public") -> dict` → `{"table","schema","columns":[...],"constraints":[...],"indexes":[...]}` or `{"error": str}` when not found.

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_schema.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_schema.py -v`
Expected: FAIL/ERROR — `AttributeError: module 'mcp_server.db' has no attribute 'list_tables'` (or SKIPPED if no DB; in that case temporarily set the URL to confirm failure, then proceed).

- [ ] **Step 3: Implement introspection**

Append to `claude-memory-compiler/mcp_server/db.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run (with `MCP_DATABASE_URL` set to the read-only role): `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_schema.py -v`
Expected: PASS. Without the URL, the tests SKIP — that is acceptable for CI, but verify against the live DB at least once locally.

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/db.py mcp_server/tests/test_schema.py
git commit -m "feat(mcp): add live schema introspection (list_tables, get_schema)"
```

---

### Task 5: Migration listing

**Files:**
- Create: `claude-memory-compiler/mcp_server/migrations.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_migrations.py`

**Interfaces:**
- Consumes: `settings.MIGRATIONS_DIR` (Task 1).
- Produces: `migrations.list_migrations(migrations_dir: Path | None = None) -> dict` → `{"files": list[str], "latest_number": int | None, "next_number": str}`.

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_migrations.py`:

```python
from mcp_server.migrations import list_migrations


def test_lists_and_computes_next(tmp_path):
    (tmp_path / "000_baseline.sql").write_text("--", encoding="utf-8")
    (tmp_path / "061_a.sql").write_text("--", encoding="utf-8")
    (tmp_path / "063_org_tax.sql").write_text("--", encoding="utf-8")
    (tmp_path / "archive").mkdir()
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    out = list_migrations(tmp_path)

    assert out["files"] == ["000_baseline.sql", "061_a.sql", "063_org_tax.sql"]
    assert out["latest_number"] == 63
    assert out["next_number"] == "064"


def test_empty_dir(tmp_path):
    out = list_migrations(tmp_path)
    assert out["files"] == []
    assert out["latest_number"] is None
    assert out["next_number"] == "000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_migrations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server.migrations'`.

- [ ] **Step 3: Implement migrations listing**

Create `claude-memory-compiler/mcp_server/migrations.py`:

```python
"""List StarBot SQL migrations and compute the next migration number."""

from __future__ import annotations

import re
from pathlib import Path

from mcp_server import settings

_NUM = re.compile(r"^(\d{3})_.*\.sql$")


def list_migrations(migrations_dir: Path | None = None) -> dict:
    """Return ordered NNN_*.sql migration files plus latest/next numbers."""
    directory = migrations_dir or settings.MIGRATIONS_DIR
    if not directory.exists():
        return {"files": [], "latest_number": None, "next_number": "000",
                "error": f"migrations dir not found: {directory}"}

    numbered: list[tuple[int, str]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = _NUM.match(entry.name)
        if m:
            numbered.append((int(m.group(1)), entry.name))

    numbered.sort()
    files = [name for _, name in numbered]
    latest = numbered[-1][0] if numbered else None
    next_num = f"{(latest + 1):03d}" if latest is not None else "000"
    return {"files": files, "latest_number": latest, "next_number": next_num}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_migrations.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/migrations.py mcp_server/tests/test_migrations.py
git commit -m "feat(mcp): add migration listing with next-number computation"
```

---

### Task 6: Knowledge search

**Files:**
- Create: `claude-memory-compiler/mcp_server/knowledge.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_knowledge.py`

**Interfaces:**
- Consumes: `settings.SCRIPTS_DIR`, `settings.KNOWLEDGE_DIR` (Task 1); existing `scripts/utils.py` functions `read_wiki_index`, `list_wiki_articles`.
- Produces:
  - `knowledge.search_knowledge(query: str, limit: int = 5) -> dict` → `{"index": str, "matches": [{"path","title","score","body"}]}`
  - `knowledge.get_knowledge_index() -> str`

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_knowledge.py`:

```python
from mcp_server import knowledge


def test_index_is_returned():
    idx = knowledge.get_knowledge_index()
    assert "Knowledge Base Index" in idx or "Article" in idx


def test_search_finds_postgrest_gotcha():
    out = knowledge.search_knowledge("postgrest or filter", limit=5)
    paths = [m["path"] for m in out["matches"]]
    assert any("postgrest-or-filter-gotcha" in p for p in paths)
    # Top results carry their full body so the caller can synthesize.
    assert out["matches"][0]["body"]
    assert out["index"]


def test_search_empty_query_returns_no_matches():
    out = knowledge.search_knowledge("", limit=5)
    assert out["matches"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_knowledge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server.knowledge'`.

- [ ] **Step 3: Implement knowledge search**

Create `claude-memory-compiler/mcp_server/knowledge.py`:

```python
"""Keyword-ranked retrieval over the compiled knowledge base.

Reuses the existing readers in scripts/utils.py and returns raw article bodies
so the *calling* agent (already an LLM) synthesizes the answer. No sub-agent
is spawned here.
"""

from __future__ import annotations

import re
import sys

from mcp_server import settings

# scripts/utils.py imports its sibling `config`, so put scripts/ on the path.
if str(settings.SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SCRIPTS_DIR))

from utils import list_wiki_articles, read_wiki_index  # noqa: E402


def get_knowledge_index() -> str:
    """Return the knowledge base index markdown."""
    return read_wiki_index()


def _score(query_terms: list[str], title: str, body: str) -> int:
    """Count term hits, weighting the title 3x."""
    title_l = title.lower()
    body_l = body.lower()
    score = 0
    for term in query_terms:
        score += 3 * len(re.findall(re.escape(term), title_l))
        score += len(re.findall(re.escape(term), body_l))
    return score


def search_knowledge(query: str, limit: int = 5) -> dict:
    """Rank knowledge articles by keyword overlap; return top bodies + index."""
    terms = [t for t in re.split(r"\s+", query.lower().strip()) if len(t) > 2]
    if not terms:
        return {"index": read_wiki_index(), "matches": []}

    scored = []
    for path in list_wiki_articles():
        body = path.read_text(encoding="utf-8")
        rel = path.relative_to(settings.KNOWLEDGE_DIR).as_posix()
        title = path.stem
        score = _score(terms, title, body)
        if score > 0:
            scored.append({"path": rel, "title": title, "score": score, "body": body})

    scored.sort(key=lambda m: m["score"], reverse=True)
    return {"index": read_wiki_index(), "matches": scored[: max(1, int(limit))]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_knowledge.py -v`
Expected: PASS (3 passed). These run against the real `knowledge/` dir, which already contains `postgrest-or-filter-gotcha.md`.

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/knowledge.py mcp_server/tests/test_knowledge.py
git commit -m "feat(mcp): add keyword knowledge search reusing utils readers"
```

---

### Task 7: Conventions and mistakes readers

**Files:**
- Create: `claude-memory-compiler/mcp_server/conventions.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_conventions.py`

**Interfaces:**
- Consumes: `settings.MISTAKES_FILE`, `settings.AGENTS_FILE` (Task 1).
- Produces:
  - `conventions.get_mistakes(topic: str | None = None) -> str`
  - `conventions.get_conventions(repo: str | None = None) -> str`

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_conventions.py`:

```python
from mcp_server import conventions


def test_get_conventions_returns_agents_md():
    text = conventions.get_conventions()
    assert "Repository Guidelines" in text or "Project Structure" in text


def test_get_mistakes_full():
    text = conventions.get_mistakes()
    assert "Mistake" in text


def test_get_mistakes_topic_filter():
    # The mistakes file documents a Modal/deferReply rule.
    text = conventions.get_mistakes("modal")
    assert "modal" in text.lower()
    # Filtering should shrink the output vs the full file.
    assert len(text) < len(conventions.get_mistakes())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_conventions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server.conventions'`.

- [ ] **Step 3: Implement readers**

Create `claude-memory-compiler/mcp_server/conventions.py`:

```python
"""Readers for workspace conventions (AGENTS.md) and the bot mistakes log."""

from __future__ import annotations

from mcp_server import settings


def _read(path) -> str:
    if not path.exists():
        return f"(not found: {path})"
    return path.read_text(encoding="utf-8")


def get_conventions(repo: str | None = None) -> str:
    """Return workspace coding conventions (root AGENTS.md).

    `repo` is accepted for forward-compatibility; v1 returns the workspace-wide
    AGENTS.md, which already documents per-repo structure and rules.
    """
    return _read(settings.AGENTS_FILE)


def get_mistakes(topic: str | None = None) -> str:
    """Return the bot 'don'ts' log, optionally filtered to blocks matching topic.

    The mistakes file is a numbered list; each entry starts with 'N)'. When a
    topic is given, return only entries whose text contains it (case-insensitive).
    """
    text = _read(settings.MISTAKES_FILE)
    if not topic:
        return text

    topic_l = topic.lower()
    # Split into numbered blocks like '\n1) ...' while keeping the header.
    import re

    blocks = re.split(r"(?=^\d+\)\s)", text, flags=re.MULTILINE)
    matched = [b for b in blocks if topic_l in b.lower()]
    if not matched:
        return f"(no mistakes entries mention '{topic}')"
    return "\n".join(matched).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_conventions.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/conventions.py mcp_server/tests/test_conventions.py
git commit -m "feat(mcp): add conventions and mistakes readers"
```

---

### Task 8: FastMCP server wiring

**Files:**
- Create: `claude-memory-compiler/mcp_server/server.py`
- Create: `claude-memory-compiler/mcp_server/tests/test_server.py`

**Interfaces:**
- Consumes: every module function from Tasks 3–7.
- Produces: `server.mcp` (FastMCP instance with 8 registered tools) and `server.main()` (stdio entrypoint that fails fast if `MCP_DATABASE_URL` is missing).

- [ ] **Step 1: Write the failing test**

Create `claude-memory-compiler/mcp_server/tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server.server'`.

- [ ] **Step 3: Implement the server**

Create `claude-memory-compiler/mcp_server/server.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd claude-memory-compiler && uv run pytest mcp_server/tests/test_server.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `cd claude-memory-compiler && uv run pytest mcp_server -v`
Expected: all non-DB tests PASS; `@needs_db` tests SKIP unless `MCP_DATABASE_URL` is set.

- [ ] **Step 6: Smoke-test the server boots and fails fast correctly**

Run (no URL): `cd claude-memory-compiler && MCP_DATABASE_URL= uv run python -c "from mcp_server import server; server.main()"`
Expected: exits with a `RuntimeError` mentioning `MCP_DATABASE_URL` (fail-fast confirmed).

- [ ] **Step 7: Commit**

```bash
cd claude-memory-compiler
git add mcp_server/server.py mcp_server/tests/test_server.py
git commit -m "feat(mcp): wire FastMCP server with all eight tools"
```

---

### Task 9: Registration, read-only role docs, and README

**Files:**
- Create: `J:/Projects/StarBot/.mcp.json`
- Create: `claude-memory-compiler/mcp_server/README.md`
- Modify: `J:/Projects/StarBot/AGENTS.md` (add a short "MCP server" pointer)

**Interfaces:**
- Consumes: the `starbot-mcp` console script (Task 1) and `server.main` (Task 8).
- Produces: client registration + operator documentation. No code/tests.

- [ ] **Step 1: Create the Claude Code project registration**

Create `J:/Projects/StarBot/.mcp.json`:

```json
{
  "mcpServers": {
    "starbot-brain": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "J:/Projects/StarBot/claude-memory-compiler",
        "starbot-mcp"
      ]
    }
  }
}
```

- [ ] **Step 2: Write the server README (incl. read-only role SQL + Codex registration)**

Create `claude-memory-compiler/mcp_server/README.md`:

```markdown
# StarBot Project-Brain MCP Server

A read-only MCP server giving every agent (Claude Code + Codex) live DB ground
truth and searchable project knowledge, to reduce repeated mistakes.

## Tools
- `list_tables`, `get_schema`, `run_select` (SELECT-only), `list_migrations`
- `search_knowledge`, `get_knowledge_index`, `get_mistakes`, `get_conventions`

## One-time setup

### 1. Create a dedicated read-only Postgres role (Supabase SQL editor)
```sql
CREATE ROLE mcp_ro WITH LOGIN PASSWORD '<strong-password>';
GRANT USAGE ON SCHEMA public TO mcp_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_ro;
ALTER ROLE mcp_ro SET statement_timeout = '5s';
ALTER ROLE mcp_ro SET default_transaction_read_only = on;
```

### 2. Configure the connection string
Copy `mcp_server/.env.example` to `mcp_server/.env` and set `MCP_DATABASE_URL`
to the `mcp_ro` connection string. Use the direct connection
(`db.<ref>.supabase.co:5432`) or the Supavisor pooler
(`...pooler.supabase.com:6543`, username `mcp_ro.<project-ref>`) depending on
IPv4/IPv6 availability. `.env` is git-ignored — never commit it.

### 3. Register with clients
- **Claude Code:** already wired via `J:/Projects/StarBot/.mcp.json`.
- **Codex:** add to `~/.codex/config.toml`:
  ```toml
  [mcp_servers.starbot-brain]
  command = "uv"
  args = ["run", "--directory", "J:/Projects/StarBot/claude-memory-compiler", "starbot-mcp"]
  ```

## Run / test
```bash
cd claude-memory-compiler
uv run starbot-mcp           # start the stdio server
uv run pytest mcp_server -v  # tests (DB tests skip without MCP_DATABASE_URL)
```

## Safety
Read-only by construction: dedicated read-only role + read-only transaction +
SELECT-only guard + 5s statement timeout + row cap (default 100, max 1000).
```

- [ ] **Step 3: Add a pointer in the workspace AGENTS.md**

Add this section near the top of `J:/Projects/StarBot/AGENTS.md` (after the intro):

```markdown
## Project-Brain MCP Server

Before guessing DB schema or repeating known gotchas, use the `starbot-brain`
MCP tools: `get_schema`/`run_select` for live DB ground truth,
`search_knowledge`/`get_mistakes` for documented patterns and don'ts, and
`list_migrations` for the next migration number. Setup: see
`claude-memory-compiler/mcp_server/README.md`.
```

- [ ] **Step 4: Verify registration loads (manual)**

Restart Claude Code in this workspace and run `/mcp` (or check the MCP panel).
Expected: `starbot-brain` appears with 8 tools. Call `list_migrations` and
confirm it returns `next_number: "064"`.

- [ ] **Step 5: Commit (compiler repo only; .mcp.json + AGENTS.md are at the non-repo workspace root)**

```bash
cd claude-memory-compiler
git add mcp_server/README.md
git commit -m "docs(mcp): add server README with read-only role and client setup"
```

Note: `J:/Projects/StarBot/.mcp.json` and `AGENTS.md` live at the workspace
root, which is **not** a git repository — they are saved in place, not
committed. Leave them as working files.

---

## Self-Review

**Spec coverage:**
- Unified server, all agents → Tasks 8–9. ✓
- `list_tables`/`get_schema`/`run_select`/`list_migrations` → Tasks 3, 4, 5, 8. ✓
- `search_knowledge`/`get_knowledge_index`/`get_mistakes`/`get_conventions` → Tasks 6, 7, 8. ✓
- Read-only role + required `MCP_DATABASE_URL` + fail-fast → Tasks 1, 8; SQL in Task 9. ✓
- 4-layer safety (role, read-only txn, SELECT guard, row cap) → Tasks 2, 3, 9. ✓
- No sub-agent spawning / reuse utils readers → Task 6. ✓
- Registration (Claude Code + Codex) → Task 9. ✓
- Testing plan (guard, ranking, conventions, migrations; gated DB tests) → Tasks 2–8. ✓
- v1 non-goal: drift-diff deferred — not implemented, as specified. ✓

**Placeholder scan:** No "TBD"/"implement later". The only angle-bracket placeholders are operator-supplied secrets (`<strong-password>`, `<ref>`, `<project-ref>`) in setup SQL/URLs, which are intentional.

**Type consistency:** `is_safe_select -> (bool, str)` used consistently (Tasks 2–3). `run_select` dict keys `columns/rows/row_count/truncated/error` consistent (Tasks 3, 8). `get_schema` keys `table/schema/columns/constraints/indexes/error` consistent (Tasks 4, 8). `list_migrations` keys `files/latest_number/next_number` consistent (Tasks 5, 8). Tool names in `server.py` match the set asserted in `test_server.py` (Task 8). ✓
