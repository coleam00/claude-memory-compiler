"""
SessionStart hook - injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index plus recent substantive Capture
Records (records/captures/, one per logical session), then injects them as
additional context so Claude always "remembers" what it has learned and what
it worked on recently.

Historical daily/*.md aggregates are read-only legacy inputs; this hook never
reads them for current context.

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
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Paths relative to project root
ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
SCRIPTS_DIR = ROOT / "scripts"
INDEX_FILE = KNOWLEDGE_DIR / "index.md"

# The shared Capture Record module lives in scripts/.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from capture_records import list_capture_records

MAX_CONTEXT_CHARS = 20_000
MAX_EXCERPT_CHARS = int(os.environ.get("KB_CAPTURE_EXCERPT_CHARS", "900"))
MAX_RECENT_RECORDS = int(os.environ.get("KB_RECENT_CAPTURES", "4"))
INDEX_DAYS = int(os.environ.get("KB_INDEX_DAYS", "14"))
INDEX_MAX_ROWS = int(os.environ.get("KB_INDEX_MAX_ROWS", "10"))
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def filter_index(content: str, cutoff: date) -> str:
    """Keep header + the INDEX_MAX_ROWS most-recently-updated article rows.

    Robust to markdown-table padding (a formatter rewrites `|---|` to `| --- |`
    and pads cells, which broke separator/line-end-date detection and silently
    dumped the entire index). A data row is any pipe-delimited line that
    contains a `[[` wikilink; its Updated date is the latest date token in the
    row. Rows are ranked newest-first and capped at INDEX_MAX_ROWS so the
    injection stays bounded. Nothing filters by date: the `cutoff` argument
    is unused, and KB_INDEX_DAYS no longer trims the index.
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


def excerpt_body(body: str) -> str:
    """Trim a record body to MAX_EXCERPT_CHARS on a line boundary."""
    if len(body) <= MAX_EXCERPT_CHARS:
        return body.strip()
    cut = body[:MAX_EXCERPT_CHARS]
    newline = cut.rfind("\n")
    if newline > MAX_EXCERPT_CHARS // 2:
        cut = cut[:newline]
    return cut.strip() + "\n\n(...excerpt truncated)"


def get_recent_captures() -> str:
    """Summarize narrative Capture Records, newest record day first.

    Sorted by (record_day, filename) descending, so same-day records order
    lexically by filename, then capped at MAX_RECENT_RECORDS.
    """
    substantive = [
        record
        for record in list_capture_records()
        if getattr(record, "capture_status", None) == "narrative"
    ]
    if not substantive:
        return "(no recent Capture Records)"

    def sort_key(record):
        day = record.record_day
        day_str = day.isoformat() if hasattr(day, "isoformat") else str(day)
        return (day_str, record.path.name)

    recent = sorted(substantive, key=sort_key, reverse=True)[:MAX_RECENT_RECORDS]
    blocks = []
    for record in recent:
        day = record.record_day
        day_str = day.isoformat() if hasattr(day, "isoformat") else str(day)
        sid = getattr(record, "source_sid", None) or "no-sid"
        blocks.append(f"### {day_str} [sid:{sid}]\n{excerpt_body(record.body)}")
    return "\n\n".join(blocks)


def build_context() -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date
    today = datetime.now(UTC).astimezone()
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

    # Recent narrative Capture Records (newest first, bounded)
    try:
        recent_captures = get_recent_captures()
    except Exception as e:  # never break session startup over context injection
        print(f"session-start: capture read failed: {e}", file=sys.stderr)
        recent_captures = "(no recent Capture Records)"
    parts.append(f"## Recent Session Captures\n\n{recent_captures}")

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
