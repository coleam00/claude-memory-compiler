"""Readers for workspace conventions (AGENTS.md) and the bot mistakes log."""

from __future__ import annotations

import re
from pathlib import Path

from mcp_server import settings


def _read(path: Path) -> str:
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
    blocks = re.split(r"(?=^\d+\)\s)", text, flags=re.MULTILINE)
    matched = [b for b in blocks if topic_l in b.lower()]
    if not matched:
        return f"(no mistakes entries mention '{topic}')"
    return "\n".join(matched).strip()
