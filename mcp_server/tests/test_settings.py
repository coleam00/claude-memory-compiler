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
