"""Regression tests for Obsidian display-alias handling in wikilink checks.

Fixes #7: ``[[concepts/my-concept|My Concept]]`` (valid Obsidian alias syntax)
was extracted verbatim, so check_broken_links looked for a file named
``concepts/my-concept|My Concept.md`` and check_orphan_pages missed inbound
links written with an alias.
"""
from __future__ import annotations

import utils
from utils import extract_wikilinks


def test_extract_wikilinks_strips_display_alias():
    assert extract_wikilinks("see [[concepts/my-concept|My Concept]] here") == [
        "concepts/my-concept",
    ]


def test_extract_wikilinks_plain_links_unchanged():
    assert extract_wikilinks("[[concepts/a]] and [[connections/b]]") == [
        "concepts/a",
        "connections/b",
    ]


def test_extract_wikilinks_alias_with_multiple_pipes():
    # A pipe inside the display text must not leak into the target.
    assert extract_wikilinks("[[qa/foo|a | b]]") == ["qa/foo"]


def test_extract_wikilinks_mixed_content():
    assert extract_wikilinks("[[a|one]] plain [[b]] [[c|two|three]]") == ["a", "b", "c"]


def test_wiki_article_exists_after_alias_strip(tmp_path, monkeypatch):
    concept = tmp_path / "concepts"
    concept.mkdir()
    (concept / "my-concept.md").write_text("body", encoding="utf-8")
    monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)
    assert utils.wiki_article_exists("concepts/my-concept") is True


def test_count_inbound_links_counts_aliased_links(tmp_path, monkeypatch):
    other = tmp_path / "other.md"
    other.write_text("[[concepts/my-concept|My Concept]]", encoding="utf-8")
    monkeypatch.setattr(utils, "list_wiki_articles", lambda: [other])
    assert utils.count_inbound_links("concepts/my-concept") == 1


def test_count_inbound_links_plain_and_aliased_both_counted(tmp_path, monkeypatch):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("[[x]]", encoding="utf-8")
    b.write_text("[[x|display]]", encoding="utf-8")
    monkeypatch.setattr(utils, "list_wiki_articles", lambda: [a, b])
    assert utils.count_inbound_links("x") == 2
