"""
Backfill `summary:` frontmatter into existing concept and connection articles
by lifting the lead paragraph (between the H1 and the next H2).

Usage:
    uv run python backfill_summary.py             # write
    uv run python backfill_summary.py --dry-run   # preview, no writes
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [
    ROOT / "knowledge" / "concepts",
    ROOT / "knowledge" / "connections",
]

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


def split_frontmatter(text: str) -> tuple[str, str] | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    return m.group(1), m.group(2)


def has_summary(frontmatter: str) -> bool:
    return bool(re.search(r"^summary:\s", frontmatter, re.MULTILINE))


def extract_lead(body: str) -> str | None:
    """Lift the first prose paragraph after the H1.

    Falls back to the first prose paragraph after any later H2 if the H1 is
    immediately followed by a section header (common in connection articles
    where the body opens with `## The Connection`).
    """
    lines = body.split("\n")
    saw_h1 = False
    after_heading = False
    buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not saw_h1:
            if line.startswith("# "):
                saw_h1 = True
                after_heading = True
            continue
        if line.startswith("## "):
            if buf:
                break
            after_heading = True
            continue
        if not after_heading:
            continue
        if not stripped:
            if buf:
                break
            continue
        if stripped.startswith(("- ", "* ", "> ", "```", "|")):
            # If the section opens with non-prose, skip ahead to the next heading
            if not buf:
                after_heading = False
                continue
            break
        buf.append(stripped)
    if not buf:
        return None
    return " ".join(buf)


def yaml_escape(value: str) -> str:
    """Quote with double quotes; escape backslashes and double quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def insert_summary(frontmatter: str, summary: str) -> str:
    """Insert `summary:` after `tags:` block (or after `title:` if no tags)."""
    summary_line = f"summary: {yaml_escape(summary)}"
    lines = frontmatter.split("\n")

    # Find the end of the tags: block (handles inline + multiline list forms)
    tags_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^tags\s*:", ln)), None
    )
    if tags_idx is not None:
        # Walk forward until we leave the tags block
        end = tags_idx + 1
        while end < len(lines):
            ln = lines[end]
            # Continuation of YAML list/value: starts with whitespace OR a `-`
            if ln.startswith((" ", "\t", "-")):
                end += 1
            else:
                break
        lines.insert(end, summary_line)
    else:
        # Fall back: after title:
        title_idx = next(
            (i for i, ln in enumerate(lines) if re.match(r"^title\s*:", ln)), 0
        )
        lines.insert(title_idx + 1, summary_line)

    return "\n".join(lines)


def process_file(path: Path, dry_run: bool) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8")
    parts = split_frontmatter(text)
    if not parts:
        return False, "no frontmatter"
    fm, body = parts
    if has_summary(fm):
        return False, "already has summary"
    lead = extract_lead(body)
    if not lead:
        return False, "no lead paragraph found"
    new_fm = insert_summary(fm, lead)
    new_text = f"---\n{new_fm}\n---\n{body}"
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True, lead[:80] + ("…" if len(lead) > 80 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(p for d in TARGET_DIRS for p in d.glob("*.md"))
    print(f"Scanning {len(files)} articles in {[d.name for d in TARGET_DIRS]}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    updated, skipped = 0, 0
    for f in files:
        changed, info = process_file(f, args.dry_run)
        rel = f.relative_to(ROOT)
        if changed:
            updated += 1
            print(f"  ✓ {rel}\n      → {info}")
        else:
            skipped += 1
            print(f"  · {rel}  ({info})")

    print(f"\n{'Would update' if args.dry_run else 'Updated'}: {updated}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
