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
load_dotenv(Path(__file__).resolve().parent / ".env")

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
