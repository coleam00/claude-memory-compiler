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
