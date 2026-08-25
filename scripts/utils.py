"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import re
from pathlib import Path

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    QA_DIR,
    STATE_FILE,
)

# ── State management ──────────────────────────────────────────────────

def load_state() -> dict:
    """Load persistent state from state.json."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


def save_state(state: dict) -> None:
    """Save state to state.json."""
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


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
    """Extract all [[wikilinks]] targets from markdown content.

    Aliased links (``[[target|Alias]]`` or the table-safe escaped form
    ``[[target\\|Alias]]``) and anchors (``#heading`` / ``#^block``) are
    stripped, leaving only the link target.
    """
    targets = []
    for raw in re.findall(r"\[\[([^\]\n]+)\]\]", content):
        target = re.split(r"\\?\|", raw, maxsplit=1)[0]
        target = target.split("#", 1)[0]
        targets.append(target.strip())
    return targets


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


def list_daily_logs() -> list[Path]:
    """List all daily log files (legacy read-only aggregate inputs)."""
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


# ── Index table normalization ─────────────────────────────────────────
# knowledge/index.md rows are LLM-authored markdown table cells. An
# unescaped `|` inside a `[[target|Alias]]` wikilink splits the row into
# extra columns, and a `` `=...` `` code span reads as a Dataview inline
# expression. These helpers repair both bug classes; `normalize_index` is
# pure and idempotent by construction.

INDEX_COLUMNS = ("Article", "Summary", "Compiled From", "Updated")


def escape_table_cell(text: str) -> str:
    """Escape every unescaped `|` in a table cell as `\\|`."""
    return re.sub(r"(?<!\\)\|", r"\\|", text)


# Dataview's inline-field reader trims a code span's text before checking
# for the inline-expression prefix `=`
# (`let text = codeblock.innerText.trim();` in dataview's src/main.ts), so
# a leading ASCII space does NOT stop a `=` from being parsed as an
# expression. U+200B (zero-width space) survives JS's `.trim()` - it is
# not in the WhiteSpace set - and renders invisibly, so it neutralizes
# the span without a visible artifact. Do not "clean up" this character.
ZERO_WIDTH_SPACE = "\u200b"


def sanitize_dataview(text: str) -> str:
    """Neutralize inline code spans Dataview would parse as an expression.

    Any single-backtick span whose content starts with `=` (optionally
    preceded by ASCII spaces, which Dataview's `.trim()` strips before
    checking the prefix) gets a U+200B zero-width space inserted right
    after the opening backtick, e.g. `` `==` `` -> `` `\u200b==` ``.
    Idempotent: a span already starting with U+200B is left alone.
    """
    def _fix(match: re.Match) -> str:
        body = match.group(1)
        if body.startswith(ZERO_WIDTH_SPACE):
            return match.group(0)
        stripped = body.lstrip(" ")
        if stripped.startswith("="):
            return f"`{ZERO_WIDTH_SPACE}{stripped}`"
        return match.group(0)

    return re.sub(r"`([^`\n]*)`", _fix, text)


def split_table_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped `|` characters.

    A `|` preceded by a backslash is treated as a literal character, not
    a column separator. The empty leading/trailing cells produced by the
    outer pipes are dropped; remaining cells are whitespace-stripped.
    """
    parts = re.split(r"(?<!\\)\|", line)
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def _is_table_separator_line(line: str) -> bool:
    """True for a `| --- | --- |`-style separator row."""
    stripped = line.strip()
    return stripped.startswith("|") and set(stripped) <= set("-:| ")


_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")


def _has_open_wikilink(cell: str) -> bool:
    """True if the cell opens a `[[` that it never closes.

    Inline code spans are ignored: prose that merely quotes ``[[`` (the
    index summarizes wikilink behaviour often enough) is not an open
    link and must not swallow the following columns.
    """
    bare = _CODE_SPAN_RE.sub("", cell)
    return bare.rfind("[[") > bare.rfind("]]")


def _repair_split_wikilinks(cells: list[str]) -> list[str]:
    """Rejoin wikilink cells that an unescaped alias pipe split apart.

    A cell that opens a wikilink (`[[...`) but never closes it (`]]`) was
    split on that link's unescaped alias pipe. Merge it with the
    following cell using the table-safe escaped pipe form, but only when
    that follower actually closes the link. An empty follower means the
    link had no alias at all, so it is simply closed.
    """
    merged: list[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        while _has_open_wikilink(cell) and i + 1 < len(cells):
            follower = cells[i + 1]
            if follower == "":
                cell = f"{cell}]]"
                i += 1
                break
            if "]]" not in _CODE_SPAN_RE.sub("", follower):
                break
            cell = f"{cell}\\|{follower}"
            i += 1
        merged.append(cell)
        i += 1
    return merged


def _normalize_data_row(line: str) -> str:
    """Repair a single knowledge/index.md data row to 4 clean columns."""
    cells = split_table_row(line)
    cells = _repair_split_wikilinks(cells)
    while cells and cells[-1] == "":
        cells.pop()
    cells = [escape_table_cell(sanitize_dataview(c)) for c in cells]
    if len(cells) < 4:
        cells = cells + [""] * (4 - len(cells))
    elif len(cells) > 4:
        cells = cells[:3] + [" ".join(cells[3:])]
    return "| " + " | ".join(cells) + " |"


def normalize_index(content: str) -> str:
    """Repair the knowledge/index.md table.

    Fixes wikilinks split by an unescaped alias pipe, drops phantom
    trailing columns, neutralizes Dataview-triggering code spans, and
    rewrites the header/separator to the canonical 4-column form.
    Leading content (the heading, blank lines) and any trailing content
    after the table are preserved verbatim. Pure function: same input
    always yields the same output, and re-normalizing already-normalized
    content is a no-op.
    """
    lines = content.split("\n")

    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        i += 1
    out: list[str] = list(lines[:i])

    if i < len(lines):
        # Header row - always rewritten to the canonical form.
        out.append("| " + " | ".join(INDEX_COLUMNS) + " |")
        i += 1
        if i < len(lines) and _is_table_separator_line(lines[i]):
            out.append("| " + " | ".join("---" for _ in INDEX_COLUMNS) + " |")
            i += 1

    while i < len(lines) and lines[i].lstrip().startswith("|"):
        out.append(_normalize_data_row(lines[i]))
        i += 1

    out.extend(lines[i:])
    return "\n".join(out)


def normalize_index_file() -> bool:
    """Repair INDEX_FILE in place. Returns True iff its content changed."""
    if not INDEX_FILE.exists():
        return False
    original = INDEX_FILE.read_text(encoding="utf-8")
    fixed = normalize_index(original)
    if fixed == original:
        return False
    INDEX_FILE.write_text(fixed, encoding="utf-8")
    return True


if __name__ == "__main__":
    changed = normalize_index_file()
    print(f"index.md {'repaired' if changed else 'already normalized'}")
