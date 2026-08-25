#!/usr/bin/env python3
"""Materialize legacy projection notes from the authoritative aggregates.

This deliberately has no dependency on the compiler runtime.  It copies each
aggregate entry verbatim after a small YAML frontmatter envelope whose date and
datetime fields remain native YAML scalars.

Each refresh updates and deletes generated legacy notes so the projection
matches the aggregates. Direct Capture Records (written by hooks, marked
`record_origin: direct`) are preserved and never touched by this projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "daily"
BUILD_LOG = ROOT / "knowledge" / "log.md"
RECORDS = ROOT / "records"
CAPTURES = RECORDS / "captures"
RUNS = RECORDS / "runs"

# This is intentionally narrow: only these canonical aggregate headings create
# records.  Generic Markdown headings are record payload, not delimiters.
CAPTURE_HEADING = re.compile(
    rb"^### (Session|Memory Flush) \(((?:backfill )?\d{2}:\d{2})\)(?: \[sid:([^\]\r\n]+)\])?(?:\r?\n)?$"
)
RUN_HEADING = re.compile(
    rb"^## \[([^\]\r\n]+)\] (compile|query_filed|query filed|query \(filed\)) \| ([^\r\n]+?)[ \t]*(?:\r?\n)?$"
)
SOURCE_LINE = re.compile(rb"^- Source: ([^\r\n]+)(?:\r?\n)?$")
DAY_PATH = re.compile(r"(?:^|/)daily/(\d{4}-\d{2}-\d{2})\.md|^(\d{4}-\d{2}-\d{2})\.md")
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]*)?\]\]")
ARTICLE_LABEL = re.compile(rb"^- Articles (created|updated):(?:[ \t]*(.*))?(?:\r?\n)?$")
FENCE_OPEN = re.compile(rb"^[ \t]{0,3}(`{3,}|~{3,})")

# Corpus drift guards.  Tests patch these to None when exercising the
# projection machinery against small fixtures.
CAPTURE_INVARIANTS: tuple[dict[str, int], int] | None = (
    {"narrative": 919, "no_op": 1601, "failure": 770},
    3290,
)
RUN_INVARIANTS: int | None = 81


@dataclass(frozen=True)
class ExpectedNote:
    path: Path
    content: bytes


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def yaml_value(value: object, native_scalar: str | None = None) -> str:
    """Emit JSON-compatible YAML except for validated native date scalars."""
    if native_scalar == "date":
        if not isinstance(value, str):
            raise TypeError(f"expected date string, got {type(value).__name__}")
        return date.fromisoformat(value).isoformat()
    if native_scalar == "datetime":
        if not isinstance(value, str):
            raise TypeError(f"expected datetime string, got {type(value).__name__}")
        return datetime.fromisoformat(value).isoformat()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def frontmatter(fields: Iterable[tuple[str, object]]) -> bytes:
    native_scalars = {
        "record_day": "date",
        "source_day": "date",
        "started_at": "datetime",
    }
    lines = ["---"]
    for key, value in fields:
        lines.append(f"{key}: {yaml_value(value, native_scalars.get(key))}")
    lines.append("---")
    return ("\n".join(lines) + "\n").encode("utf-8")


def lines_with_offsets(data: bytes) -> list[tuple[int, bytes]]:
    offset = 0
    result: list[tuple[int, bytes]] = []
    for line in data.splitlines(keepends=True):
        result.append((offset, line))
        offset += len(line)
    if data and not result:  # Defensive, though splitlines covers nonempty data.
        result.append((0, data))
    return result


def capture_boundaries(lines: list[tuple[int, bytes]]) -> list[tuple[int, re.Match[bytes]]]:
    """Find exact capture headings, ignoring headings inside fenced code blocks."""
    boundaries: list[tuple[int, re.Match[bytes]]] = []
    fence_char: int | None = None
    fence_length = 0
    for index, (_, line) in enumerate(lines):
        fence = FENCE_OPEN.match(line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                # A closing fence admits only trailing whitespace.
                trailing = line[fence.end() :].rstrip(b"\r\n")
                if not trailing.strip(b" \t"):
                    fence_char = None
                    fence_length = 0
            continue
        if fence_char is None:
            match = CAPTURE_HEADING.match(line)
            if match:
                boundaries.append((index, match))
    return boundaries


def capture_status(kind: str, payload: bytes) -> str:
    # A Session is a narrative by contract. Memory Flush status is taken only
    # from its explicit marker; it is never inferred from prose or headings.
    if kind == "Session":
        return "narrative"
    content = [
        line.rstrip(b"\r\n") for line in payload.splitlines(keepends=True)[1:] if line.strip()
    ]
    if content == [b"FLUSH_OK - Nothing worth saving from this session"]:
        return "no_op"
    if content and content[0].startswith(b"FLUSH_ERROR"):
        return "failure"
    if content:
        return "narrative"
    raise ValueError("empty Memory Flush record")


def capture_notes() -> list[ExpectedNote]:
    notes: list[ExpectedNote] = []
    statuses = {"narrative": 0, "no_op": 0, "failure": 0}
    for daily_file in sorted(DAILY.glob("*.md")):
        source = daily_file.read_bytes()
        source_hash = sha256(source)
        lines = lines_with_offsets(source)
        boundaries = capture_boundaries(lines)
        day = daily_file.stem
        source_file = f"daily/{daily_file.name}"
        records: list[tuple[int, int, bytes, re.Match[bytes]]] = []
        tuple_totals: dict[tuple[str, str, str, str], int] = {}
        for ordinal, (start_index, match) in enumerate(boundaries, start=1):
            end_index = boundaries[ordinal][0] if ordinal < len(boundaries) else len(lines)
            payload = source[
                lines[start_index][0] : lines[end_index][0]
                if end_index < len(lines)
                else len(source)
            ]
            kind = match.group(1).decode("utf-8")
            timestamp = match.group(2).decode("utf-8")
            sid = match.group(3).decode("utf-8") if match.group(3) is not None else None
            capture_kind = "session" if kind == "Session" else "memory_flush"
            identity = sid or sha256(payload)[:16]
            key = (day, capture_kind, timestamp, identity)
            tuple_totals[key] = tuple_totals.get(key, 0) + 1
            records.append((start_index, end_index, payload, match))

        tuple_seen: dict[tuple[str, str, str, str], int] = {}
        for ordinal, (start_index, end_index, payload, match) in enumerate(records, start=1):
            kind = match.group(1).decode("utf-8")
            timestamp = match.group(2).decode("utf-8")
            sid = match.group(3).decode("utf-8") if match.group(3) is not None else None
            capture_kind = "session" if kind == "Session" else "memory_flush"
            identity = sid or sha256(payload)[:16]
            key = (day, capture_kind, timestamp, identity)
            tuple_seen[key] = tuple_seen.get(key, 0) + 1
            occurrence = tuple_seen[key]
            record_tuple = key + ((occurrence,) if tuple_totals[key] > 1 else ())
            record_id = "capture:" + json.dumps(
                record_tuple, ensure_ascii=False, separators=(",", ":")
            )
            heading_kind = capture_kind.replace("_", "-")
            stamp_slug = (
                "backfill-" + timestamp.removeprefix("backfill ").replace(":", "")
                if timestamp.startswith("backfill ")
                else "live-" + timestamp.replace(":", "")
            )
            suffix = f"--{occurrence}" if tuple_totals[key] > 1 else ""
            filename = (
                f"{day}--{heading_kind}--{stamp_slug}--{quote(identity, safe='-._~')}{suffix}.md"
            )
            status = capture_status(kind, payload)
            statuses[status] += 1
            fields = (
                ("record_type", "capture_record"),
                ("record_id", record_id),
                ("record_day", day),
                ("capture_kind", capture_kind),
                ("capture_status", status),
                ("source_timestamp", timestamp),
                ("source_sid", sid),
                ("source_file", source_file),
                ("source_aggregate_file", source_file),
                ("source_ordinal", ordinal),
                ("source_line_start", start_index + 1),
                ("source_line_end", end_index),
                ("payload_sha256", sha256(payload)),
                ("source_aggregate_sha256", source_hash),
                ("tags", ["memory-compiler/capture-record"]),
            )
            notes.append(ExpectedNote(CAPTURES / day / filename, frontmatter(fields) + payload))
    if CAPTURE_INVARIANTS is not None:
        statuses_expected, total_expected = CAPTURE_INVARIANTS
        if statuses != statuses_expected:
            raise ValueError(
                f"unexpected capture status counts: {statuses!r} (expected {statuses_expected!r})"
            )
        if len(notes) != total_expected:
            raise ValueError(
                f"unexpected capture record count: {len(notes)} (expected {total_expected})"
            )
    return notes


_DIRECT_ORIGIN_RE = re.compile(rb"^record_origin:[ \t]*\"?direct\"?[ \t\r]*$", re.MULTILINE)


def is_direct_note(path: Path) -> bool:
    """True when a note is a hook-written direct Capture Record."""
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if not data.startswith(b"---\n"):
        return False
    close = data.find(b"\n---", 4)
    block = data[: close] if close != -1 else data
    return _DIRECT_ORIGIN_RE.search(block) is not None


def normalize_source_path(value: str) -> tuple[str, str]:
    """Reduce legacy bare/prefixed source spellings to the one identity path."""
    match = DAY_PATH.search(value.strip())
    if not match:
        raise ValueError(f"could not find daily source path in {value!r}")
    day = match.group(1) or match.group(2)
    return f"daily/{day}.md", day


def articles_in(lines: list[bytes]) -> tuple[list[str], list[str]]:
    created: list[str] = []
    updated: list[str] = []
    destination: list[str] | None = None
    for line in lines:
        label = ARTICLE_LABEL.match(line)
        if label:
            destination = created if label.group(1) == b"created" else updated
            fragments = [label.group(2) or b""]
        elif destination is not None and line.startswith((b" ", b"\t")):
            fragments = [line]
        else:
            destination = None
            continue
        for fragment in fragments:
            for target in WIKILINK.findall(fragment.decode("utf-8")):
                destination.append(target)
    return created, updated


def run_qualifier(header_source: str) -> str | None:
    """Keep only the optional qualifier attached to the heading source operand."""
    match = re.search(r"(?:^|/)daily/\d{4}-\d{2}-\d{2}\.md|^\d{4}-\d{2}-\d{2}\.md", header_source)
    if not match:
        raise ValueError(f"could not find daily source path in heading {header_source!r}")
    tail = header_source[match.end() :].strip()
    if not tail:
        return None
    if tail.startswith("(") and tail.endswith(")"):
        return tail[1:-1]
    return tail


def normalize_operation(value: str) -> str:
    """Map supported run-heading verbs to their stable record operation."""
    if value == "query (filed)":
        return "query_filed"
    return value.replace(" ", "_")


def run_notes() -> list[ExpectedNote]:
    source = BUILD_LOG.read_bytes()
    source_hash = sha256(source)
    lines = lines_with_offsets(source)
    boundaries: list[tuple[int, re.Match[bytes]]] = []
    for index, (_, line) in enumerate(lines):
        match = RUN_HEADING.match(line)
        if match:
            boundaries.append((index, match))
    run_id_totals: dict[tuple[str, str, str], int] = {}
    for _, match in boundaries:
        operation = normalize_operation(match.group(2).decode("utf-8"))
        source_day = normalize_source_path(match.group(3).decode("utf-8"))[1]
        key = (operation, match.group(1).decode("utf-8"), source_day)
        run_id_totals[key] = run_id_totals.get(key, 0) + 1

    run_id_seen: dict[tuple[str, str, str], int] = {}

    notes: list[ExpectedNote] = []
    for ordinal, (start_index, match) in enumerate(boundaries, start=1):
        end_index = boundaries[ordinal][0] if ordinal < len(boundaries) else len(lines)
        start_offset = lines[start_index][0]
        end_offset = lines[end_index][0] if end_index < len(lines) else len(source)
        payload = source[start_offset:end_offset]
        started_at = match.group(1).decode("utf-8")
        operation = normalize_operation(match.group(2).decode("utf-8"))
        header_source = match.group(3).decode("utf-8")
        source_line_value: str | None = None
        for _, line in lines[start_index + 1 : end_index]:
            source_match = SOURCE_LINE.match(line)
            if source_match:
                source_line_value = source_match.group(1).decode("utf-8")
                break
        source_file, source_day = normalize_source_path(source_line_value or header_source)
        qualifier = run_qualifier(header_source)
        created, updated = articles_in([line for _, line in lines[start_index + 1 : end_index]])
        if created and updated:
            change_kind = "created_and_updated"
        elif created:
            change_kind = "created_only"
        elif updated:
            change_kind = "updated_only"
        else:
            change_kind = "no_article_change"
        run_key = (operation, started_at, source_day)
        run_id_seen[run_key] = run_id_seen.get(run_key, 0) + 1
        occurrence = run_id_seen[run_key]
        timestamp_slug = started_at.replace(":", "-")
        record_id = f"{operation}__{timestamp_slug}__{source_day}"
        if run_id_totals[run_key] > 1:
            record_id = f"{record_id}__{occurrence}"
        filename = f"{record_id}.md"
        fields = (
            ("record_type", "compilation_run"),
            ("record_id", record_id),
            ("operation", operation),
            ("started_at", started_at),
            ("source_file", source_file),
            ("source_day", source_day),
            ("source_aggregate_file", "knowledge/log.md"),
            ("run_qualifier", qualifier),
            ("change_kind", change_kind),
            ("articles_created", created),
            ("articles_updated", updated),
            ("source_line_start", start_index + 1),
            ("source_line_end", end_index),
            ("payload_sha256", sha256(payload)),
            ("source_aggregate_sha256", source_hash),
            ("tags", ["memory-compiler/compilation-run"]),
        )
        notes.append(ExpectedNote(RUNS / source_day / filename, frontmatter(fields) + payload))
    if RUN_INVARIANTS is not None and len(notes) != RUN_INVARIANTS:
        raise ValueError(f"unexpected compilation run count: {len(notes)} (expected {RUN_INVARIANTS})")
    return notes


def expected_notes() -> dict[Path, bytes]:
    records = capture_notes() + run_notes()
    expected = {note.path: note.content for note in records}
    if len(expected) != len(records):
        raise ValueError("record filename collision")
    return expected


def generated_notes(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    result: set[Path] = set()
    for path in root.rglob("*.md"):
        # Do not claim or delete a hand-authored note merely because it is
        # located underneath an owned root.
        if path.read_bytes().startswith(b"---\nrecord_type: "):
            result.add(path)
    return result


def differences(expected: dict[Path, bytes]) -> tuple[list[Path], list[Path], list[Path]]:
    # Direct Capture Records are hook-authored state; exclude them from every
    # missing/changed/stale computation so refresh never touches them.
    actual = {
        path
        for path in generated_notes(CAPTURES) | generated_notes(RUNS)
        if not is_direct_note(path)
    }
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual)
    changed = sorted(
        path for path in expected_paths & actual if path.read_bytes() != expected[path]
    )
    stale = sorted(actual - expected_paths)
    return missing, changed, stale


def write_bytes_atomic(path: Path, content: bytes) -> None:
    """Replace path with content via a same-directory temp file + os.replace.

    Readers never observe a partial file: the temp file is fully written and
    fsynced before the atomic rename, and the temp file is removed if any
    step fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
        raise


def relative(paths: Iterable[Path]) -> str:
    return ", ".join(str(path.relative_to(ROOT)) for path in paths)


def refresh() -> tuple[int, int, int, int]:
    """Write the projection; returns (total, added, updated, stale removed)."""
    expected = expected_notes()
    missing, changed, stale = differences(expected)
    for path, content in expected.items():
        if path.exists():
            existing = path.read_bytes()
            if existing == content or is_direct_note(path):
                continue  # Current, or a direct record claimed this path.
        write_bytes_atomic(path, content)
    for path in stale:
        path.unlink()
        parent = path.parent
        while (
            parent in (CAPTURES, RUNS)
            or parent.is_relative_to(CAPTURES)
            or parent.is_relative_to(RUNS)
        ):
            if parent in (CAPTURES, RUNS) or any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent
    return len(expected), len(missing), len(changed), len(stale)


def refresh_projection() -> None:
    """Best-effort refresh for aggregate writers; never raises.

    The aggregates stay authoritative: a failed refresh only leaves the
    records/ projection stale until the next successful run.
    """
    try:
        refresh()
    except Exception as error:  # noqa: BLE001 - refresh must not break a writer
        print(f"materialize_records: projection refresh failed: {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report expected output mismatches without writing"
    )
    args = parser.parse_args()

    if args.check:
        expected = expected_notes()
        missing, changed, stale = differences(expected)
        if missing or changed or stale:
            for label, paths in (("missing", missing), ("changed", changed), ("stale", stale)):
                if paths:
                    print(f"{label} ({len(paths)}): {relative(paths)}")
            return 1
        print(f"records are current: {len(expected)} notes")
        return 0

    total, added, updated, removed = refresh()
    print(f"materialized {total} notes ({added} added, {updated} updated, {removed} stale removed)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"materialize_records: {error}", file=sys.stderr)
        raise SystemExit(1)
