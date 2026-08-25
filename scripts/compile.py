"""Compile pending Capture Records into knowledge articles via OMP agents.

Each substantive (narrative) Capture Record is compiled by exactly one OMP
agent. The prompt names only the record and its same-day Raw Input paths; the
agent reads the knowledge index and relevant articles itself, so article
bodies never travel through argv or stdin.

Usage:
    uv run python scripts/compile.py                # compile pending records
    uv run python scripts/compile.py --all          # recompile every narrative Capture Record
    uv run python scripts/compile.py --file records/captures/2026-08-24/x.md
    uv run python scripts/compile.py --dry-run      # show assignments only
"""

from __future__ import annotations

# Recursion prevention: set BEFORE any imports that might trigger an agent.
# Without this, a spawned OMP subprocess's own SessionEnd hook fires at exit
# -> flush.py -> another compile.py, looping.
import os

os.environ["CLAUDE_INVOKED_BY"] = "knowledge_compiler"

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from capture_records import (
    CaptureRecord,
    capture_state_key,
    discover_unprocessed_raw,
    list_capture_records,
    read_capture,
    tag_raw_inputs,
)
from config import COMPILE_LOCK_FILE as LOCK_FILE
from config import COMPILE_MODELS, ROOT_DIR, STATE_FILE, now_iso
from materialize_records import refresh_projection
from utils import load_state, normalize_index_file

# ── Compilation contract constants ────────────────────────────────────
PROMPT_PREFIX = "/skill:unslop"
LEGACY_BOOTSTRAP_MODEL = "legacy"
SUBSTANTIVE_STATUS = "narrative"
DIAGNOSTIC_TAIL_LINES = 20

_FRONTMATTER_FIELD_RE = re.compile(r'^([a-z0-9_]+):\s*"?(.*?)"?\s*$', re.MULTILINE)


def sha256_file(path: Path) -> str:
    """Full SHA-256 hexdigest of a file's bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frontmatter(path: Path) -> dict[str, str]:
    """Minimal frontmatter key/value read for fields CaptureRecord omits."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    return dict(_FRONTMATTER_FIELD_RE.findall(text[3:end]))


def _day_str(record_day) -> str:
    return str(record_day)[:10]


# ── Selection ─────────────────────────────────────────────────────────


def select_pending(
    records: tuple[CaptureRecord, ...], state: dict, *, force: bool = False
) -> list[CaptureRecord]:
    """Sorted substantive records that still need compilation.

    A record is pending when it is narrative and either never compiled or its
    payload changed since the recorded compile (changed-payload retry).
    """
    compiled = state.get("compiled_captures", {})
    pending = []
    for record in sorted(records, key=lambda r: (_day_str(r.record_day), r.path.name)):
        if record.capture_status != SUBSTANTIVE_STATUS:
            continue
        if not force:
            previous = compiled.get(capture_state_key(record))
            if previous and previous.get("payload_sha256") == record.payload_sha256:
                continue
        pending.append(record)
    return pending


def assign_models(pending: list[CaptureRecord]) -> list[tuple[CaptureRecord, str]]:
    """Round-robin model assignment over sorted records; counts differ by ≤1."""
    return [(record, COMPILE_MODELS[i % len(COMPILE_MODELS)]) for i, record in enumerate(pending)]


def plan_raw_attachments(
    pending: list[CaptureRecord], state: dict
) -> dict[str, tuple[Path, ...]]:
    """Attach a day's unprocessed Raw Inputs to that day's first pending record only."""
    attachments: dict[str, tuple[Path, ...]] = {
        capture_state_key(record): () for record in pending
    }
    seen_days: set[str] = set()
    for record in pending:
        day = _day_str(record.record_day)
        if day in seen_days:
            continue
        seen_days.add(day)
        attachments[capture_state_key(record)] = discover_unprocessed_raw(day, state)
    return attachments


def bootstrap_legacy_compiles(records: tuple[CaptureRecord, ...], state: dict) -> int:
    """Mark historical records compiled when their aggregate hash matches legacy state.

    Legacy whole-file daily hashes live in ``state['ingested']``. Every record
    whose ``source_aggregate_sha256`` starts with the matching daily hash was
    already part of a compiled aggregate; bootstrapping them prevents replaying
    already compiled history on the first post-cutover run.
    """
    ingested = state.get("ingested", {})
    compiled = state.setdefault("compiled_captures", {})
    bootstrapped = 0
    for record in records:
        key = capture_state_key(record)
        if key in compiled:
            continue
        frontmatter = _read_frontmatter(record.path)
        aggregate_sha = frontmatter.get("source_aggregate_sha256", "")
        source_name = Path(
            frontmatter.get("source_aggregate_file")
            or frontmatter.get("source_file")
            or f"daily/{_day_str(record.record_day)}.md"
        ).name
        legacy_hash = ingested.get(source_name, {}).get("hash")
        if not (aggregate_sha and legacy_hash and aggregate_sha.startswith(legacy_hash)):
            continue
        compiled[key] = {
            "payload_sha256": record.payload_sha256,
            "compiled_at": now_iso(),
            "model": LEGACY_BOOTSTRAP_MODEL,
        }
        bootstrapped += 1
    return bootstrapped


# ── Prompt / command ──────────────────────────────────────────────────


def build_prompt(record: CaptureRecord, raw_paths: tuple[Path, ...]) -> str:
    """Prompt naming exactly one Capture Record plus Raw Input paths."""
    lines = [
        PROMPT_PREFIX,
        "",
        "You are a knowledge compiler working inside this Obsidian vault.",
        "Compile exactly one Capture Record into knowledge articles.",
        "",
        "## Capture Record",
        "",
        str(record.path),
        "",
        "## Same-day Raw Inputs",
        "",
    ]
    if raw_paths:
        lines.extend(str(path) for path in raw_paths)
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "## Procedure",
            "",
            "1. Read knowledge/index.md first to see what the wiki already covers.",
            "2. Read the Capture Record listed above.",
            "3. Read only the existing knowledge articles relevant to this record.",
            "4. Write or update articles following the AGENTS.md schema, then update",
            "   knowledge/index.md and append a timestamped entry to knowledge/log.md.",
            "",
            "## Rules",
            "",
            "- Index table rows have exactly 4 cells; escape alias pipes as \\| and never",
            "  start an inline code span with =.",
            "- Every new article needs complete YAML frontmatter and [[wikilinks]] to at",
            "  least 2 other articles.",
        ]
    )
    return "\n".join(lines)


def build_command(model: str, prompt: str) -> list[str]:
    """The exact OMP invocation; the prompt is paths only, never article bodies."""
    return [
        "omp",
        "-p",
        "--no-session",
        "--auto-approve",
        "--cwd",
        str(ROOT_DIR),
        "--model",
        model,
        prompt,
    ]


def invoke_omp(command: list[str]) -> tuple[bool, str]:
    """Run one OMP agent, capturing stdout/stderr for diagnostics."""
    environment = dict(os.environ)
    environment["CLAUDE_INVOKED_BY"] = "knowledge_compiler"
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output


# ── Locking / state persistence ───────────────────────────────────────


@contextmanager
def compile_lock():
    """One global interprocess lock: state load through saves run strictly
    sequentially, so no compile can snapshot ahead of the lock."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "w")  # noqa: SIM115 - closed in finally below
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def save_state_atomic(state: dict) -> None:
    """Atomic replace so a crash never leaves a torn state.json."""
    descriptor, temp_path = tempfile.mkstemp(
        dir=STATE_FILE.parent, prefix=".state-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(temp_path, STATE_FILE)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


# ── Per-record compilation ────────────────────────────────────────────


def _relative_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def compile_record(
    record: CaptureRecord,
    model: str,
    raw_paths: tuple[Path, ...],
    state: dict,
) -> bool:
    """Invoke one OMP agent; persist success-only state/tag transitions."""
    key = capture_state_key(record)
    print(f"\n[{model}] Compiling {record.path.name} ({len(raw_paths)} raw inputs)...")

    if raw_paths:
        # Claim the inputs before invocation so a concurrent compiler cannot
        # double-attach them; compiled tags land only after success.
        tag_raw_inputs(raw_paths, compiled=False, capture_id=key)

    ok, output = invoke_omp(build_command(model, build_prompt(record, raw_paths)))
    if not ok:
        tail = "\n".join(output.splitlines()[-DIAGNOSTIC_TAIL_LINES:])
        print(f"  Failed.\n{tail}", file=sys.stderr)
        return False

    timestamp = now_iso()
    state.setdefault("compiled_captures", {})[key] = {
        "payload_sha256": record.payload_sha256,
        "compiled_at": timestamp,
        "model": model,
    }
    if raw_paths:
        # Final compiled tags land first so the stored hash covers exactly
        # the tagged bytes on disk; discovery excludes a raw only when its
        # current sha256 matches, so hashing pre-tag bytes would re-attach
        # every already-compiled input on the next run.
        tag_raw_inputs(raw_paths, compiled=True, capture_id=key)
    raw_compiled = state.setdefault("compiled_raw_inputs", {})
    for path in raw_paths:
        if not path.exists():
            continue
        raw_compiled[_relative_key(path)] = {
            "sha256": sha256_file(path),
            "compiled_at": timestamp,
            "capture_id": key,
        }
    save_state_atomic(state)

    if normalize_index_file():
        print("  Repaired index.md formatting.")
    refresh_projection()
    print("  Done.")
    return True


# ── CLI ───────────────────────────────────────────────────────────────


def _resolve_record_file(argument: str) -> Path:
    target = Path(argument)
    candidates = [target] if target.is_absolute() else [Path.cwd() / target, ROOT_DIR / target]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print(f"Error: {argument} not found")
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile pending Capture Records into knowledge articles"
    )
    parser.add_argument("--all", action="store_true", help="Recompile every narrative Capture Record")
    parser.add_argument("--file", type=str, help="Compile one specific Capture Record file")
    parser.add_argument("--dry-run", action="store_true", help="Show assignments without compiling")
    args = parser.parse_args(argv)

    failed: list[str] = []
    with compile_lock():
        # The lock is held from the first state read through the last save so
        # two serialized compiles can never load a snapshot ahead of the lock
        # and replay work against stale state. Dry runs share the same locked
        # snapshot but never mutate anything.
        state = load_state()
        records = list_capture_records()

        if args.file:
            target = _resolve_record_file(args.file)
            record = read_capture(target)
            if record.capture_status != SUBSTANTIVE_STATUS:
                name = target.name
                status = record.capture_status
                print(f"Skipping {name}: capture_status={status} (not substantive)")
                return 0
            pending = [record]
        else:
            bootstrapped = bootstrap_legacy_compiles(records, state)
            if bootstrapped:
                print(f"Bootstrapped {bootstrapped} historical Capture Records as compiled.")
                if not args.dry_run:
                    save_state_atomic(state)
            pending = select_pending(records, state, force=args.all)

        if not pending:
            print("Nothing to compile - all substantive Capture Records are up to date.")
            return 0

        assignments = assign_models(pending)
        attachments = {} if args.file else plan_raw_attachments(pending, state)

        label = "[DRY RUN] " if args.dry_run else ""
        print(f"{label}Records to compile ({len(assignments)}):")
        for record, model in assignments:
            raw_count = len(attachments.get(capture_state_key(record), ()))
            print(f"  - {record.path.name} -> {model} (+{raw_count} raw)")
        if args.dry_run:
            return 0

        for record, model in assignments:
            ok = compile_record(
                record, model, attachments.get(capture_state_key(record), ()), state
            )
            if not ok:
                failed.append(record.path.name)

    if failed:
        print("\nCompilation failed for: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"\nCompilation complete: {len(assignments)} records compiled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
