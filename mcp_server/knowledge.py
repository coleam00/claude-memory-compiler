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
