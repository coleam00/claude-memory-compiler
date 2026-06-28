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


def test_missing_dir(tmp_path):
    out = list_migrations(tmp_path / "does_not_exist")
    assert out["files"] == []
    assert out["latest_number"] is None
    assert out["next_number"] == "000"
    assert "error" in out
