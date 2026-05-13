"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    LOG_FILE,
    QA_DIR,
    STATE_FILE,
)


def _default_state() -> dict:
    """Fresh default state. Always returns a NEW dict (with fresh nested
    containers) so callers can mutate without leaking into later calls."""
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


# ── State management ──────────────────────────────────────────────────

def load_state() -> dict:
    """Load persistent state from state.json.

    Resilient to empty or truncated files: if the file exists but cannot
    be parsed as JSON (zero bytes, partial write from a previous crash,
    etc.), fall back to the default state and log a warning instead of
    raising. This prevents a single corrupted state file from stalling
    every subsequent compile run.
    """
    if not STATE_FILE.exists():
        return _default_state()
    raw = STATE_FILE.read_text(encoding="utf-8")
    if not raw.strip():
        logging.warning(
            "state.json exists but is empty (likely a truncated write); "
            "starting from default state"
        )
        return _default_state()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.warning(
            "state.json is not valid JSON (%s); starting from default state. "
            "The bad file has been preserved as state.json.broken for inspection.",
            exc,
        )
        try:
            STATE_FILE.rename(STATE_FILE.with_suffix(".json.broken"))
        except OSError:
            pass
        return _default_state()


def save_state(state: dict) -> None:
    """Save state to state.json atomically.

    Writes to a temp file in the same directory and uses os.replace() so
    a crash mid-write leaves either the previous good file or the new
    good file, never a truncated/empty one. The previous implementation
    used Path.write_text which is non-atomic and corrupted state.json on
    crashes — silently disabling every subsequent compile run.
    """
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


# ── File hashing ──────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Slug / naming ─────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


# ── Wikilink helpers ──────────────────────────────────────────────────

def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from markdown content."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


def wiki_article_exists(link: str) -> bool:
    """Check if a wikilinked article exists on disk."""
    path = KNOWLEDGE_DIR / f"{link}.md"
    return path.exists()


# ── Wiki content helpers ──────────────────────────────────────────────

def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "# Knowledge Base Index\n\n| Article | Summary | Compiled From | Updated |\n|---------|---------|---------------|---------|"


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if not subdir.exists():
            continue
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(KNOWLEDGE_DIR)
            content = md_file.read_text(encoding="utf-8")
            parts.append(f"## {rel}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def list_wiki_articles() -> list[Path]:
    """List all wiki article files."""
    articles = []
    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if subdir.exists():
            articles.extend(sorted(subdir.glob("*.md")))
    return articles


def list_raw_files() -> list[Path]:
    """List all daily log files."""
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("*.md"))


# ── Index helpers ─────────────────────────────────────────────────────

def count_inbound_links(target: str, exclude_file: Path | None = None) -> int:
    """Count how many wiki articles link to a given target."""
    count = 0
    for article in list_wiki_articles():
        if article == exclude_file:
            continue
        content = article.read_text(encoding="utf-8")
        if f"[[{target}]]" in content:
            count += 1
    return count


def get_article_word_count(path: Path) -> int:
    """Count words in an article, excluding YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]
    return len(content.split())


def build_index_entry(rel_path: str, summary: str, sources: str, updated: str) -> str:
    """Build a single index table row."""
    link = rel_path.replace(".md", "")
    return f"| [[{link}]] | {summary} | {sources} | {updated} |"
