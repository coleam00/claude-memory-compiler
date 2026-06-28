# StarBot Project-Brain MCP Server

A read-only MCP server giving every agent (Claude Code + Codex) live DB ground
truth and searchable project knowledge, to reduce repeated mistakes.

## Tools
- `list_tables`, `get_schema`, `run_select` (SELECT-only), `list_migrations`
- `search_knowledge`, `get_knowledge_index`, `get_mistakes`, `get_conventions`

## One-time setup

### 1. Create a dedicated read-only Postgres role (Supabase SQL editor)
```sql
CREATE ROLE mcp_ro WITH LOGIN PASSWORD '';
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
