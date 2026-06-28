# Project-Brain MCP Server — Design

**Date:** 2026-06-28
**Status:** Approved (design), pending implementation plan
**Location of code:** `claude-memory-compiler/mcp_server/`

## 1. Purpose

A single stdio MCP server that **every agent** (Claude Code and Codex, across the
`StarBot`, `dashboard`, `nomads`, and `library` repos) can call to check *ground
truth* instead of guessing. It directly attacks the two recurring classes of agent
mistake documented in this workspace's own knowledge base:

1. **Schema-guessing / migration drift** — agents assume column names/types and get
   burned (see `knowledge/concepts/live-schema-ground-truth.md`,
   `connections/migration-drift-and-silent-failures.md`). The server exposes the
   **live** schema and guarded read-only queries.
2. **Repeating documented gotchas** — agents re-make mistakes already captured in the
   knowledge base and `.claude/mistakes.md` (e.g. the PostgREST `.or()` gotcha, dual
   permission systems, dayjs timezone pitfalls). The server makes that knowledge
   queryable before an agent acts.

## 2. Goals / Non-Goals

**Goals**
- One server, all agents, all repos in the workspace.
- Live DB **ground truth**: schema/metadata + SELECT-only read access.
- Searchable project knowledge + conventions + mistakes.
- Cheap, deterministic, fast — **no Anthropic API calls** at runtime (the calling
  agent is already an LLM; the server hands it raw facts to reason over).
- Read-only by construction; cannot mutate the database.

**Non-Goals (v1)**
- No writes to the DB. No applying migrations (user keeps manual SQL control).
- No semantic/vector search — keyword + index ranking is sufficient at current scale
  (Karpathy/no-RAG rationale already adopted by `query.py`).
- No spawning of sub-agents (unlike `query.py`'s `run_query`, which uses the Claude
  Agent SDK — deliberately *not* reused, to keep tool calls cheap and synchronous).
- Not a public/HTTP service. stdio only; launched locally by each agent client.

## 3. Architecture

```
claude-memory-compiler/
  mcp_server/
    __init__.py
    server.py        # FastMCP app, tool registration, stdio entrypoint
    db.py            # psycopg connection, SQL guard, schema introspection
    knowledge.py     # reuses utils.read_all_wiki_content / read_wiki_index + ranking
    conventions.py   # readers for AGENTS.md, .claude/mistakes.md, per-repo CLAUDE.md
    settings.py      # paths + MCP_DATABASE_URL loading (fail-fast, never logged)
  tests/
    test_sql_guard.py
    test_knowledge.py
    test_conventions.py
    test_migrations.py
  pyproject.toml     # add deps: mcp, psycopg[binary]
```

- **Transport:** stdio (standard local MCP).
- **Entrypoint:** console script `starbot-mcp` → `mcp_server.server:main`, run via
  `uv run starbot-mcp`.
- **Framework:** official MCP Python SDK (`mcp`), `FastMCP` for ergonomic tool defs.
- **Reuse:** `knowledge.py` imports `read_all_wiki_content`, `read_wiki_index`,
  `list_wiki_articles` from the existing `scripts/utils.py`; it does **not** reuse
  `query.py`'s agent-spawning path.

### Data flow
```
agent → MCP client (stdio) → server.py tool
      → db.py | knowledge.py | conventions.py
      → JSON result → agent
```

## 4. Tools

### Ground truth (live DB, read-only)

| Tool | Input | Output | Notes |
|------|-------|--------|-------|
| `list_tables` | `schema="public"` | `[{table, estimated_rows}]` | From `pg_catalog`/`pg_class.reltuples`. |
| `get_schema` | `table`, `schema="public"` | columns (name, type, nullable, default, identity, position), primary key, foreign keys, indexes, check constraints | From `information_schema` + `pg_catalog`. The flagship anti-drift tool. |
| `run_select` | `sql`, `limit=100` | `{columns, rows, row_count, truncated}` | SELECT-only, guarded, read-only txn (see §6). |
| `list_migrations` | — | `{files:[...], latest_number, next_number}` | Lists `StarBot/src/scripts/migrations/` in numeric order; computes the **next migration number** to use. |

### Knowledge / conventions (files)

| Tool | Input | Output | Notes |
|------|-------|--------|-------|
| `search_knowledge` | `query`, `limit=5` | `{index, matches:[{path, title, score, body}]}` | Keyword/substring ranking over `knowledge/` articles; returns full bodies + the index so the caller synthesizes. |
| `get_knowledge_index` | — | `index.md` text | Cheap overview of all articles. |
| `get_mistakes` | `topic=None` | `.claude/mistakes.md` (full or keyword-filtered sections) | The bot-specific "don'ts" list. |
| `get_conventions` | `repo=None` | Relevant `AGENTS.md` (workspace root) and, if present, the repo's `CLAUDE.md`/`AGENTS.md` | Coding style, commit rules, dashboard mutation rules. |

## 5. Configuration & DB credential

- The server **requires** `MCP_DATABASE_URL` (the connection string for a dedicated
  **read-only** Postgres role). If unset, the server **fails fast at startup** with a
  clear message — no silent fallback to a writable connection
  (`knowledge/concepts/crypto-key-fail-fast.md`).
- `MCP_DATABASE_URL` is loaded from the server's own `.env`
  (`claude-memory-compiler/mcp_server/.env`, git-ignored) or the process environment.
  It is **never logged and never returned** in any tool result.
- Workspace paths (knowledge dir, migrations dir, mistakes/AGENTS files) are derived
  relative to the compiler root in `settings.py`.

### One-time read-only role setup (user runs manually in Supabase SQL editor)
```sql
CREATE ROLE mcp_ro WITH LOGIN PASSWORD '<strong-password>';
GRANT USAGE ON SCHEMA public TO mcp_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_ro;
ALTER ROLE mcp_ro SET statement_timeout = '5s';
ALTER ROLE mcp_ro SET default_transaction_read_only = on;
```
`information_schema`/`pg_catalog` are readable by `PUBLIC` by default, so `mcp_ro` can
introspect. Connection string form to confirm at implementation time (direct
`db.<ref>.supabase.co:5432` vs Supavisor pooler `…pooler.supabase.com:6543` with the
`mcp_ro.<project-ref>` username, depending on IPv4/IPv6 availability).

## 6. Safety model (defense in depth)

`run_select` is read-only by **four** independent layers:

1. **DB role** — `mcp_ro` has only `SELECT`; physically cannot write or DDL.
2. **Read-only transaction** — each query runs in `BEGIN; SET TRANSACTION READ ONLY;
   SET LOCAL statement_timeout='5s';` and the transaction is rolled back.
3. **SQL parse guard** — accepts a single statement that is `SELECT` or
   `WITH … SELECT`; rejects multiple statements (`;`-chaining), comments used to
   smuggle statements, and any `INSERT/UPDATE/DELETE/MERGE/DDL/COPY/GRANT/CALL/DO`
   token. Rejection returns a clear explanatory error.
4. **Row cap** — results truncated to `limit` (default 100, hard max e.g. 1000);
   `truncated:true` flagged when more rows existed.

Credentials are never echoed. Tool errors are structured, never stack-crashes.

## 7. Error handling

- DB unreachable / auth fail → structured error: "cannot connect (read-only role);
  check MCP_DATABASE_URL".
- Guard rejection → error naming the disallowed construct and that only `SELECT` is
  allowed.
- Statement timeout → "query exceeded 5s; narrow it or add filters".
- Missing knowledge dir / mistakes / AGENTS file → error naming the expected path.
- Unknown table in `get_schema` → empty/`not found`, not an exception.

## 8. Testing (TDD)

- `test_sql_guard.py`: accepts valid `SELECT`/`WITH…SELECT`; rejects INSERT, UPDATE,
  DELETE, DROP, ALTER, multi-statement, comment-smuggled statements, `;` chaining.
- `test_knowledge.py`: ranking returns the expected article for a known query (e.g.
  "postgrest or filter" → `postgrest-or-filter-gotcha.md`); index returned.
- `test_conventions.py`: readers return root AGENTS.md content; topic filter on
  mistakes works.
- `test_migrations.py`: numeric ordering + `next_number` computed from fixture files.
- DB introspection: covered by an integration test gated on `MCP_DATABASE_URL`
  presence (skipped in CI without it); guard logic is unit-tested without a DB.

## 9. Dependencies

Add to `claude-memory-compiler/pyproject.toml`: `mcp` (Python SDK), `psycopg[binary]`.
`python-dotenv` if not already present (compiler may already load env).

## 10. Registration

- **Claude Code:** workspace `J:/Projects/StarBot/.mcp.json` (project-scoped, so every
  Claude Code agent opened in this workspace gets the server):
  ```json
  {
    "mcpServers": {
      "starbot-brain": {
        "command": "uv",
        "args": ["run", "--directory",
                 "J:/Projects/StarBot/claude-memory-compiler", "starbot-mcp"]
      }
    }
  }
  ```
- **Codex:** equivalent `mcp_servers` entry in `~/.codex/config.toml` (or the
  workspace `.codex/` config), same command/args.

## 11. Open questions / future follow-ups

- True schema↔migrations **drift diff** (compare live columns against the SQL in
  migration files) — v2; v1 only lists migrations + next number.
- Whether to also index `dashboard`/`nomads` per-repo docs in `search_knowledge`.
- Optional `get_table_sample(table, limit)` convenience wrapper over `run_select`.
- Confirm direct-vs-pooler connection string at implementation time.
