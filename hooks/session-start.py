"""
SessionStart hook - injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index and recent daily log, then injects
them as additional context so Claude always "remembers" what it has learned.

Configure in .claude/settings.json:
{
    "hooks": {
        "SessionStart": [{
            "matcher": "",
            "command": "uv run python hooks/session-start.py"
        }]
    }
}
"""

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Paths relative to project root
ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
DAILY_DIR = ROOT / "daily"
INDEX_FILE = KNOWLEDGE_DIR / "index.md"

MAX_CONTEXT_CHARS = 20_000
MAX_LOG_LINES = 30
INDEX_DAYS = int(os.environ.get("KB_INDEX_DAYS", "14"))
INDEX_MAX_ROWS = int(os.environ.get("KB_INDEX_MAX_ROWS", "10"))
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def filter_index(content: str, cutoff: date) -> str:
    """Keep header + the INDEX_MAX_ROWS most-recently-updated article rows.

    Robust to markdown-table padding (a formatter rewrites `|---|` to `| --- |`
    and pads cells, which broke the old separator/line-end-date detection and
    silently dumped the entire index). A data row is any pipe-delimited line that
    contains a `[[` wikilink; its Updated date is the latest date token in the
    row. Rows are ranked newest-first and capped so the injection stays bounded.
    `cutoff` is retained only for the section label.
    """
    header, rows = [], []
    seen_data = False
    for line in content.splitlines():
        stripped = line.strip()
        is_data = stripped.startswith("|") and "[[" in line
        if is_data:
            seen_data = True
            dates = [date.fromisoformat(d) for d in DATE_RE.findall(line)]
            rows.append((max(dates) if dates else date.min, line))
        elif not seen_data:
            header.append(line)  # title, column header, separator — keep as-is
    if not rows:
        return content  # fallback: show full index if parsing found no rows
    rows.sort(key=lambda r: r[0], reverse=True)
    top = [line for _, line in rows[:INDEX_MAX_ROWS]]
    return "\n".join(header + top)


def get_recent_log() -> str:
    """Read the most recent daily log (today or yesterday)."""
    today = datetime.now(timezone.utc).astimezone()

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = DAILY_DIR / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            # Return last N lines to keep context small
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def build_context() -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date
    today = datetime.now(timezone.utc).astimezone()
    parts.append(f"## Today\n{today.strftime('%A, %B %d, %Y')}")

    # Knowledge base index (the core retrieval mechanism)
    if INDEX_FILE.exists():
        index_content = INDEX_FILE.read_text(encoding="utf-8")
        cutoff = today.date() - timedelta(days=INDEX_DAYS)
        filtered = filter_index(index_content, cutoff)
        parts.append(
            f"## Knowledge Base Index ({INDEX_MAX_ROWS} most recently updated)\n\n{filtered}"
        )
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    # Recent daily log
    recent_log = get_recent_log()
    parts.append(f"## Recent Daily Log\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    # Truncate if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def main():
    context = build_context()

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
