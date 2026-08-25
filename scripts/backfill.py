"""
Backfill daily logs from historical Claude Code transcripts.

Walks ~/.claude/projects/*/*.jsonl, groups messages by (local-date, session_id),
and runs flush.py with --date for each group so historical conversations get
captured into the right daily log. After all flushes, optionally runs compile.py.

Usage:
    uv run python scripts/backfill.py --date 2026-04-20
    uv run python scripts/backfill.py --from 2026-04-14 --to 2026-04-20
    uv run python scripts/backfill.py --date 2026-04-20 --dry-run
    uv run python scripts/backfill.py --date 2026-04-20 --projects '*whisper*'
    uv run python scripts/backfill.py --date 2026-04-20 --no-compile

Cost estimate (printed before running): ~$0.03 per session flush, ~$0.50 per day compile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, date as date_cls, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
PROJECTS_DIR = Path.home() / ".claude" / "projects"

MAX_TURNS_PER_SESSION = 60
MAX_CONTEXT_CHARS = 30_000
MIN_TURNS_TO_FLUSH = 2

EST_FLUSH_COST = 0.03
EST_COMPILE_COST = 0.50


def parse_date(s: str) -> date_cls:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date_cls, end: date_cls):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def iter_top_level_transcripts(project_glob: str | None):
    if not PROJECTS_DIR.exists():
        return
    for proj in sorted(PROJECTS_DIR.iterdir()):
        if not proj.is_dir():
            continue
        if project_glob and not proj.match(project_glob):
            continue
        for jsonl in sorted(proj.glob("*.jsonl")):
            yield proj.name, jsonl


def extract_dated_messages(transcript: Path, target_dates: set[date_cls]) -> dict[date_cls, list[str]]:
    """Return {date: [formatted_turn, ...]} for user/assistant messages on target dates."""
    out: dict[date_cls, list[str]] = defaultdict(list)

    try:
        f = transcript.open(encoding="utf-8")
    except OSError:
        return out

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_str = entry.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            local_date = ts.astimezone().date()
            if local_date not in target_dates:
                continue

            msg = entry.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = entry.get("role", "")
                content = entry.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts)

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                out[local_date].append(f"**{label}:** {content.strip()}\n")

    return out


def trim_context(turns: list[str]) -> str:
    recent = turns[-MAX_TURNS_PER_SESSION:]
    ctx = "\n".join(recent)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[-MAX_CONTEXT_CHARS:]
        boundary = ctx.find("\n**")
        if boundary > 0:
            ctx = ctx[boundary + 1 :]
    return ctx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="Single date YYYY-MM-DD")
    g.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD (use with --to)")
    p.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--projects", help="Glob pattern to filter project directories (e.g. '*whisper*')")
    p.add_argument("--dry-run", action="store_true", help="Show inventory + cost without flushing")
    p.add_argument("--no-compile", action="store_true", help="Skip compile.py after flushing")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--max-cost", type=float, default=20.0, help="Abort if estimate exceeds this (USD)")
    args = p.parse_args()

    if args.date:
        target_dates = {parse_date(args.date)}
    else:
        if not args.to_date:
            p.error("--to is required when using --from")
        target_dates = set(daterange(parse_date(args.from_date), parse_date(args.to_date)))

    print(f"Scanning {PROJECTS_DIR} for transcripts touching {len(target_dates)} date(s)...")

    # (date, session_id, project) -> [turns]
    grouped: dict[tuple[date_cls, str, str], list[str]] = defaultdict(list)
    transcripts_scanned = 0

    for proj_name, transcript in iter_top_level_transcripts(args.projects):
        transcripts_scanned += 1
        per_date = extract_dated_messages(transcript, target_dates)
        if not per_date:
            continue
        session_id = transcript.stem
        for d, turns in per_date.items():
            grouped[(d, session_id, proj_name)].extend(turns)

    if not grouped:
        print(f"No matching messages found across {transcripts_scanned} transcripts.")
        return 0

    # Print inventory grouped by date
    by_date: dict[date_cls, list[tuple[str, str, int]]] = defaultdict(list)
    for (d, sid, proj), turns in grouped.items():
        if len(turns) < MIN_TURNS_TO_FLUSH:
            continue
        by_date[d].append((sid, proj, len(turns)))

    if not by_date:
        print(f"All sessions had < {MIN_TURNS_TO_FLUSH} turns. Nothing to flush.")
        return 0

    # Durable dedup: skip sessions already recorded in their target daily log.
    # flush.py now writes "[sid:<session_id>]" into each section heading, so a
    # substring match against the daily log is sufficient.
    skipped_dedup = 0
    for d in list(by_date.keys()):
        log_path = DAILY_DIR / f"{d.isoformat()}.md"
        if not log_path.exists():
            continue
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except OSError:
            continue
        kept: list[tuple[str, str, int]] = []
        for sid, proj, n_turns in by_date[d]:
            if f"[sid:{sid}]" in log_text or f"[sid:backfill-{d.isoformat()}-{sid}]" in log_text:
                skipped_dedup += 1
            else:
                kept.append((sid, proj, n_turns))
        if kept:
            by_date[d] = kept
        else:
            del by_date[d]

    if skipped_dedup:
        print(f"\n(skipping {skipped_dedup} sessions already recorded in target daily log)")

    if not by_date:
        print("All sessions already flushed. Nothing to do.")
        return 0

    total_sessions = sum(len(rows) for rows in by_date.values())
    est_flushes = total_sessions * EST_FLUSH_COST
    est_compiles = len(by_date) * EST_COMPILE_COST if not args.no_compile else 0
    est_total = est_flushes + est_compiles

    print(f"\nScanned {transcripts_scanned} transcripts. Inventory:\n")
    for d in sorted(by_date):
        rows = sorted(by_date[d], key=lambda r: -r[2])
        print(f"  {d.isoformat()}  ({len(rows)} sessions, {sum(r[2] for r in rows)} turns total)")
        for sid, proj, n_turns in rows[:10]:
            print(f"    - {proj[:60]:60s}  {sid[:8]}  {n_turns} turns")
        if len(rows) > 10:
            print(f"    ... and {len(rows) - 10} more")

    print(f"\nEstimated cost: ${est_flushes:.2f} flushes + ${est_compiles:.2f} compiles = ~${est_total:.2f}")

    if est_total > args.max_cost:
        print(f"ABORT: estimate ${est_total:.2f} exceeds --max-cost ${args.max_cost:.2f}")
        return 2

    if args.dry_run:
        print("\n(--dry-run: no flushes executed)")
        return 0

    if not args.yes:
        try:
            resp = input("\nProceed? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    # Run flushes sequentially (avoid parallel SDK calls)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    flush_script = SCRIPTS_DIR / "flush.py"
    successes, failures = 0, 0

    for d in sorted(by_date):
        for sid, proj, _ in by_date[d]:
            turns = grouped[(d, sid, proj)]
            ctx = trim_context(turns)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            ctx_file = SCRIPTS_DIR / f"backfill-{d.isoformat()}-{sid[:8]}-{stamp}.md"
            ctx_file.write_text(ctx, encoding="utf-8")

            print(f"  flushing {d.isoformat()} {sid[:8]} ({proj[:40]})...", end=" ", flush=True)
            # Date-scoped session id so dedup only catches same-day same-sid duplicates
            # (e.g. cross-project mirrored project dirs), not different days.
            cmd = [
                "uv", "run", "--directory", str(ROOT), "python", str(flush_script),
                str(ctx_file), f"backfill-{d.isoformat()}-{sid}",
                "--date", d.isoformat(),
                "--no-compile",
            ]
            try:
                # Use unique session id per (date, sid) so dedup doesn't kill repeated flushes
                env = os.environ.copy()
                subprocess.run(cmd, check=True, env=env, capture_output=True, timeout=180)
                successes += 1
                print("ok")
            except subprocess.CalledProcessError as e:
                failures += 1
                print(f"FAILED: {e.stderr.decode('utf-8', errors='replace')[:200]}")
            except subprocess.TimeoutExpired:
                failures += 1
                print("TIMEOUT")

    print(f"\nFlushes: {successes} ok, {failures} failed")

    if args.no_compile:
        print("Skipping compile (--no-compile).")
        return 0 if failures == 0 else 1

    print("\nRunning compile.py...")
    compile_script = SCRIPTS_DIR / "compile.py"
    try:
        subprocess.run(
            ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"compile.py failed: {e}")
        return 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
