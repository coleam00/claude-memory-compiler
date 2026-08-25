"""Direct Capture Record storage: one Markdown note per capture.

Hooks (PreCompact, SessionEnd) call `write_capture_record`; the compiler
consumes records via `list_capture_records`.  Identity is the session ID
when the event carries one, and a payload-hash prefix when it does not.
Identity is stable for `(record_day, source_cwd, identity)` across hook
events, so overlapping events resolve to one path and the later write
atomically replaces the earlier payload.  Historical materialized records
under the same tree parse unchanged.

All writes take an interprocess lock and land via atomic replace, so a reader
never observes a torn file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from config import (
    CAPTURE_LOCK_FILE,
    CAPTURE_RECORDS_DIR,
    LOCAL_TZ,
    RAW_DIR,
)

CAPTURE_RECORD_TAG = "memory-compiler/capture-record"
RAW_INPUT_TAG = "memory-compiler/raw-input"
RAW_COMPILED_TAG = "memory-compiler/compiled"

_DIRECT_IDENTITY = "direct"
_CAPTURE_KINDS = ("session", "memory_flush")
_CAPTURE_STATUSES = ("narrative", "no_op", "failure")

_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):(.*)$", re.DOTALL)
_CREATED_RE = re.compile(r"^created:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_DAY_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class CaptureRecord:
    path: Path
    record_id: str
    record_day: str
    capture_kind: str
    capture_status: str
    payload_sha256: str
    source_sid: str | None
    body: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Locking and atomic writes ─────────────────────────────────────────


@contextmanager
def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


# ── Frontmatter primitives ────────────────────────────────────────────


def _frontmatter(fields: list[tuple[str, object]]) -> bytes:
    lines = ["---"]
    for key, value in fields:
        if key == "record_day":
            # Native YAML date scalar, matching historical materialized notes.
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _split_frontmatter(path: Path, text: str) -> tuple[list[str], str]:
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter")
    lines = text.splitlines(keepends=True)
    if lines[0].rstrip("\r\n") != "---":
        raise ValueError(f"{path}: malformed frontmatter opener")
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n") == "---":
            return [line.rstrip("\r\n") for line in lines[1:index]], "".join(lines[index + 1 :])
    raise ValueError(f"{path}: unterminated frontmatter")


def _parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value.startswith(("[", "{")) or (value.startswith('"') and value.endswith('"')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value in ("", "null", "~"):
        return None
    return value


# ── Direct record writing ─────────────────────────────────────────────


def direct_identity(session_id: str | None, payload_sha256: str) -> str:
    """Session ID when present; otherwise a content-derived fallback."""
    return session_id if session_id else payload_sha256[:16]


def capture_record_id(record_day: str, source_cwd: str | None, identity: str) -> str:
    project = source_cwd or ""
    return (
        "direct:"
        + json.dumps([record_day, project, identity], ensure_ascii=False, separators=(",", ":"))
    )


def capture_record_path(
    record_day: str, source_cwd: str | None, identity: str
) -> Path:
    project_slug = re.sub(r"[^A-Za-z0-9._~-]+", "-", (source_cwd or "").rsplit("/", 1)[-1])
    project_slug = project_slug.strip("-") or "unknown-project"
    filename = (
        f"{record_day}--{_DIRECT_IDENTITY}"
        f"--{quote(project_slug, safe='-._~')}"
        f"--{quote(identity, safe='-._~')}.md"
    )
    return CAPTURE_RECORDS_DIR / record_day / filename


def write_capture_record(
    content: str,
    *,
    capture_kind: str,
    capture_status: str,
    session_id: str | None,
    source_cwd: str | None,
    captured_at: datetime,
    record_day: str | None = None,
) -> Path:
    """Write (or atomically overwrite) the direct Capture Record for one identity."""
    if capture_kind not in _CAPTURE_KINDS:
        raise ValueError(f"unknown capture_kind: {capture_kind!r}")
    if capture_status not in _CAPTURE_STATUSES:
        raise ValueError(f"unknown capture_status: {capture_status!r}")

    local_captured = captured_at.astimezone(LOCAL_TZ)
    day = record_day or local_captured.date().isoformat()
    payload_hash = sha256_hex(content.encode("utf-8"))
    identity = direct_identity(session_id, payload_hash)
    record_id = capture_record_id(day, source_cwd, identity)
    path = capture_record_path(day, source_cwd, identity)

    fields: list[tuple[str, object]] = [
        ("record_type", "capture_record"),
        ("record_id", record_id),
        ("record_day", day),
        ("capture_kind", capture_kind),
        ("capture_status", capture_status),
        ("source_timestamp", local_captured.isoformat(timespec="seconds")),
        ("source_sid", session_id),
        ("source_project", source_cwd or ""),
        ("payload_sha256", payload_hash),
        ("record_origin", _DIRECT_IDENTITY),
        ("tags", [CAPTURE_RECORD_TAG]),
    ]
    document = _frontmatter(fields) + content.encode("utf-8")

    with _locked(CAPTURE_LOCK_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, document)
    return path


# ── Reading ───────────────────────────────────────────────────────────


def read_capture(path: Path) -> CaptureRecord:
    text = path.read_text(encoding="utf-8")
    front_lines, body = _split_frontmatter(path, text)
    fields = {
        match.group(1): _parse_scalar(match.group(2))
        for line in front_lines
        if (match := _FIELD_RE.match(line))
    }
    missing = {"record_id", "record_day", "capture_status", "payload_sha256"} - set(fields)
    if missing:
        raise ValueError(f"{path}: capture record missing fields {sorted(missing)}")
    return CaptureRecord(
        path=path,
        record_id=str(fields["record_id"]),
        record_day=str(fields["record_day"]),
        capture_kind=str(fields.get("capture_kind", "")),
        capture_status=str(fields["capture_status"]),
        payload_sha256=str(fields["payload_sha256"]),
        source_sid=fields.get("source_sid"),  # type: ignore[arg-type]
        body=body,
    )


def list_capture_records() -> tuple[CaptureRecord, ...]:
    """All parseable Capture Records, sorted by path."""
    if not CAPTURE_RECORDS_DIR.exists():
        return ()
    records: list[CaptureRecord] = []
    for path in sorted(CAPTURE_RECORDS_DIR.rglob("*.md")):
        try:
            records.append(read_capture(path))
        except (ValueError, OSError, UnicodeDecodeError):
            continue  # Non-record notes under the tree are not ours to fail on.
    return tuple(records)


def capture_state_key(record: CaptureRecord) -> str:
    """State key under `compiled_captures` for a compiled record."""
    return record.record_id


# ── Raw Input discovery and tagging ───────────────────────────────────


def _raw_created_day(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    block = text[: end if end != -1 else len(text)]
    created = _CREATED_RE.search(block)
    if not created:
        return None
    day = _DAY_PREFIX_RE.match(created.group(1).strip())
    return day.group(1) if day else None


def discover_unprocessed_raw(record_day: str, state: dict) -> tuple[Path, ...]:
    """Same-day Raw Inputs whose current SHA-256 was never compiled."""
    compiled_hashes = {
        entry.get("sha256")
        for entry in (state.get("compiled_raw_inputs") or {}).values()
        if isinstance(entry, dict)
    }
    found: list[Path] = []
    for path in sorted(RAW_DIR.glob("*.md")):
        if _raw_created_day(path) != record_day:
            continue
        if sha256_hex(path.read_bytes()) in compiled_hashes:
            continue
        found.append(path)
    return tuple(found)


def _parse_tag_values(raw: bytes) -> list[str]:
    """Read a flow-style YAML sequence, tolerating unquoted scalars."""
    text = raw.strip().decode("utf-8", errors="replace")
    if not (text.startswith("[") and text.endswith("]")):
        return []
    try:
        return [str(item) for item in json.loads(text)]
    except json.JSONDecodeError:
        inner = text[1:-1]
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]


def _apply_raw_tags(data: bytes, add_tags: list[str], compiled_in: str | None) -> bytes:
    """Merge tags/compiled_in into frontmatter; everything else stays verbatim.

    Accepts both LF and CRLF files: the existing newline style is preserved
    and a file that already has YAML never gets a second block prepended.
    """
    compiled_line = (
        None if compiled_in is None else b"compiled_in: " + json.dumps(compiled_in).encode()
    )

    lines = data.splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        header = [b"---", b"tags: " + json.dumps(add_tags, ensure_ascii=False).encode("utf-8")]
        if compiled_line is not None:
            header.append(compiled_line)
        header += [b"---", b""]
        return b"\n".join(header) + data

    # Locate the exact closing delimiter by content, tolerating CRLF.
    close = next(
        (index for index in range(1, len(lines)) if lines[index].rstrip(b"\r\n") == b"---"),
        len(lines),
    )
    eol = b"\r\n" if close < len(lines) and lines[close].endswith(b"\r\n") else b"\n"

    existing: list[str] = []
    kept: list[bytes] = []
    skip_block_items = False
    for line in lines[1:close]:
        if skip_block_items:
            if line.startswith(b" ") or line.startswith(b"\t"):
                continue  # Block-sequence item of a removed property.
            skip_block_items = False
        if line.startswith(b"tags:") or line.startswith(b"compiled_in:"):
            existing.extend(_parse_tag_values(line.split(b":", 1)[1]))
            skip_block_items = True
            continue
        kept.append(line)

    merged = list(dict.fromkeys(existing + add_tags))
    additions = [b"tags: " + json.dumps(merged, ensure_ascii=False).encode("utf-8") + eol]
    if compiled_line is not None:
        additions.append(compiled_line + eol)
    # Kept lines carry their original terminators; only new lines take `eol`.
    return b"".join([lines[0], *kept, *additions, *lines[close:]])


def tag_raw_inputs(paths: tuple[Path, ...], *, compiled: bool, capture_id: str | None = None) -> None:
    """Tag Raw Inputs; preserve every other YAML property and the body verbatim."""
    tags = [RAW_INPUT_TAG] + ([RAW_COMPILED_TAG] if compiled else [])
    compiled_in = capture_id if (compiled and capture_id) else None
    with _locked(CAPTURE_LOCK_FILE):
        for path in paths:
            data = path.read_bytes()
            updated = _apply_raw_tags(data, tags, compiled_in)
            if updated != data:
                _atomic_write(path, updated)
