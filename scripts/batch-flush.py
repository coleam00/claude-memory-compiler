"""
Batch extraction of knowledge from historical Claude Code transcripts.

Reads JSONL transcript files, extracts conversation content with smart
chunking for large sessions, runs LLM extraction on each chunk, groups
results by date, and writes daily logs ready for compile.py.

Usage:
    uv run python scripts/batch-flush.py --dry-run           # preview what would be processed
    uv run python scripts/batch-flush.py                      # run full extraction
    uv run python scripts/batch-flush.py --compile            # extract + compile
    uv run python scripts/batch-flush.py --max-cost 5.00      # stop after $5 spent
    uv run python scripts/batch-flush.py --resume             # skip already-processed sessions
    uv run python scripts/batch-flush.py --dates 2026-04-11   # only specific dates
"""

from __future__ import annotations

# Recursion prevention — set before any imports that might trigger Claude
import os
os.environ["CLAUDE_INVOKED_BY"] = "batch_flush"

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_FILE = SCRIPTS_DIR / "state.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"

def default_transcripts_dir() -> Path:
    """Auto-discover the transcripts directory from the current working directory.

    Claude Code encodes a project's absolute path in the directory name by
    replacing ``/`` with ``-``. For a project at ``/Users/foo/bar``, transcripts
    live under ``~/.claude/projects/-Users-foo-bar/``.
    """
    encoded = str(Path.cwd().resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded


DEFAULT_TRANSCRIPTS_DIR = default_transcripts_dir()

MIN_FILE_SIZE = 5_000           # Skip files < 5KB
CHUNK_TARGET_CHARS = 25_000     # Target chunk size for LLM extraction
FLUSH_COST_ESTIMATE = 0.04     # Estimated cost per flush call

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [batch] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Also log to stderr for interactive use
console = logging.StreamHandler(sys.stderr)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TranscriptInfo:
    path: Path
    size: int
    mtime: datetime
    date: str               # YYYY-MM-DD from mtime
    session_id: str         # UUID from filename

@dataclass
class Turn:
    role: str               # "user" or "assistant"
    text: str
    index: int

@dataclass
class Chunk:
    text: str
    char_count: int
    position: str           # "early", "mid", "late", "full"
    turn_range: tuple[int, int] = (0, 0)

@dataclass
class Extraction:
    session_id: str
    date: str
    time_str: str
    chunk_position: str
    content: str
    cost: float = 0.0


# ── Transcript scanning ─────────────────────────────────────────────────

def scan_transcripts(transcripts_dir: Path) -> list[TranscriptInfo]:
    """Scan all top-level JSONL transcript files (skip subagent files)."""
    results = []
    if not transcripts_dir.exists():
        logging.error("Transcripts directory not found: %s", transcripts_dir)
        return results

    for f in transcripts_dir.glob("*.jsonl"):
        # Skip if inside a subagents directory
        if "subagents" in str(f):
            continue
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
        session_id = f.stem  # UUID
        results.append(TranscriptInfo(
            path=f,
            size=stat.st_size,
            mtime=mtime,
            date=mtime.strftime("%Y-%m-%d"),
            session_id=session_id,
        ))

    results.sort(key=lambda t: t.mtime)
    return results


# ── Conversation extraction ──────────────────────────────────────────────

def extract_full_conversation(transcript_path: Path) -> list[Turn]:
    """Parse ALL user/assistant text turns from a JSONL transcript."""
    turns: list[Turn] = []
    index = 0

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
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

            # Extract text from content blocks, skip tool_use/tool_result/image blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                text = "\n".join(text_parts)
            elif isinstance(content, str):
                text = content
            else:
                continue

            text = text.strip()
            if text:
                turns.append(Turn(role=role, text=text, index=index))
                index += 1

    return turns


def extract_tool_summary(transcript_path: Path) -> str:
    """Extract a brief summary of tool usage (files edited, commands run)."""
    edited_files: set[str] = set()
    commands: list[str] = []

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name in ("Edit", "Write") and isinstance(tool_input, dict):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        # Shorten to relative path
                        fp = fp.replace("/Users/alvin/assethub/assethub-web/", "")
                        edited_files.add(fp)
                elif tool_name == "Bash" and isinstance(tool_input, dict):
                    cmd = tool_input.get("command", "")
                    if cmd and len(cmd) < 100:
                        commands.append(cmd)

    parts = []
    if edited_files:
        files_list = sorted(edited_files)[:15]  # Cap at 15
        parts.append(f"Files edited: {', '.join(files_list)}")
    if commands:
        # Show unique command prefixes
        unique_cmds = list(dict.fromkeys(cmd.split()[0] for cmd in commands if cmd.split()))[:10]
        parts.append(f"Commands used: {', '.join(unique_cmds)}")

    return "; ".join(parts) if parts else ""


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_conversation(turns: list[Turn], target_chars: int = CHUNK_TARGET_CHARS) -> list[Chunk]:
    """Split conversation into chunks at user-message boundaries."""
    if not turns:
        return []

    # Format all turns as markdown
    formatted = []
    for t in turns:
        label = "User" if t.role == "user" else "Assistant"
        formatted.append((t.index, f"**{label}:** {t.text}\n"))

    # Calculate total text
    total_chars = sum(len(text) for _, text in formatted)

    # If small enough, single chunk
    if total_chars <= target_chars * 1.3:
        text = "\n".join(text for _, text in formatted)
        return [Chunk(
            text=text,
            char_count=len(text),
            position="full",
            turn_range=(turns[0].index, turns[-1].index),
        )]

    # Split into chunks at user-message boundaries
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_chars = 0
    chunk_start_idx = turns[0].index

    for i, (turn_idx, text) in enumerate(formatted):
        # Check if adding this turn exceeds the target and we're at a user boundary
        is_user_turn = turns[i].role == "user"
        if current_chars >= target_chars and is_user_turn and current_texts:
            # Flush current chunk
            chunk_text = "\n".join(current_texts)
            chunks.append(Chunk(
                text=chunk_text,
                char_count=len(chunk_text),
                position="",  # assigned later
                turn_range=(chunk_start_idx, turns[i - 1].index),
            ))
            current_texts = []
            current_chars = 0
            chunk_start_idx = turn_idx

        current_texts.append(text)
        current_chars += len(text)

    # Flush remaining
    if current_texts:
        chunk_text = "\n".join(current_texts)
        chunks.append(Chunk(
            text=chunk_text,
            char_count=len(chunk_text),
            position="",
            turn_range=(chunk_start_idx, turns[-1].index),
        ))

    # Assign position labels
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        if n == 1:
            chunk.position = "full"
        elif i == 0:
            chunk.position = "early"
        elif i == n - 1:
            chunk.position = "late"
        else:
            chunk.position = "mid"

    return chunks


# ── LLM flush ────────────────────────────────────────────────────────────

async def flush_chunk(chunk_text: str, session_meta: str) -> str:
    """Call Claude Agent SDK to extract knowledge from one chunk."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the daily log.
Do NOT use any tools — just return plain text.

{session_meta}

Format your response as a structured daily log entry with these sections:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Important Q&A or discussions]

**Decisions Made:**
- [Any decisions with rationale]

**Lessons Learned:**
- [Gotchas, patterns, or insights discovered]

**Action Items:**
- [Follow-ups or TODOs mentioned]

Skip anything that is:
- Routine tool calls or file reads
- Content that's trivial or obvious
- Trivial back-and-forth or clarification exchanges

Only include sections that have actual content. If nothing is worth saving,
respond with exactly: FLUSH_OK

## Conversation Context

{chunk_text}"""

    response = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT),
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage):
                pass
    except Exception as e:
        import traceback
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response


# ── Daily log writing ────────────────────────────────────────────────────

def write_daily_logs(extractions: list[Extraction]) -> list[Path]:
    """Group extractions by date and write to daily log files."""
    grouped: dict[str, list[Extraction]] = {}
    for ext in extractions:
        grouped.setdefault(ext.date, []).append(ext)

    written: list[Path] = []
    for date_str in sorted(grouped.keys()):
        log_path = DAILY_DIR / f"{date_str}.md"
        DAILY_DIR.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            log_path.write_text(
                f"# Daily Log: {date_str}\n\n## Sessions\n\n",
                encoding="utf-8",
            )

        with open(log_path, "a", encoding="utf-8") as f:
            for ext in grouped[date_str]:
                if "FLUSH_OK" in ext.content:
                    continue  # Skip empty extractions
                if "FLUSH_ERROR" in ext.content:
                    f.write(f"### Session ({ext.time_str}) - Error\n\n{ext.content}\n\n")
                    continue

                position_label = f" [{ext.chunk_position}]" if ext.chunk_position != "full" else ""
                f.write(f"### Session ({ext.time_str}){position_label}\n\n{ext.content}\n\n")

        written.append(log_path)

    return written


# ── State management ─────────────────────────────────────────────────────

def load_batch_state() -> dict:
    """Load batch flush state from state.json."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return state.get("batch_flush", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed_sessions": {}, "total_cost": 0.0}


def save_batch_state(batch_state: dict) -> None:
    """Save batch flush state to state.json."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    else:
        state = {}

    state["batch_flush"] = batch_state
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────

async def run_batch(args: argparse.Namespace) -> None:
    """Main batch processing loop."""
    transcripts_dir = Path(args.transcripts_dir).expanduser()
    transcripts = scan_transcripts(transcripts_dir)

    if not transcripts:
        logging.info("No transcripts found in %s", transcripts_dir)
        return

    # Filter by date if specified
    if args.dates:
        date_set = set(args.dates.split(","))
        transcripts = [t for t in transcripts if t.date in date_set]

    # Filter by size
    skipped_tiny = [t for t in transcripts if t.size < MIN_FILE_SIZE]
    transcripts = [t for t in transcripts if t.size >= MIN_FILE_SIZE]

    # Load state for resume
    batch_state = load_batch_state()
    processed = batch_state.get("processed_sessions", {})

    if args.resume:
        before = len(transcripts)
        transcripts = [t for t in transcripts if t.session_id not in processed]
        logging.info("Resume: skipping %d already-processed sessions", before - len(transcripts))

    # Collect dates
    dates = sorted(set(t.date for t in transcripts))

    # Print summary
    logging.info("")
    logging.info("=== Batch Flush Summary ===")
    logging.info("Transcripts dir: %s", transcripts_dir)
    logging.info("Total sessions found: %d", len(transcripts) + len(skipped_tiny))
    logging.info("Skipping %d tiny sessions (<5KB)", len(skipped_tiny))
    logging.info("Processing: %d sessions across %d dates", len(transcripts), len(dates))
    logging.info("Date range: %s to %s", dates[0] if dates else "n/a", dates[-1] if dates else "n/a")
    logging.info("")

    # Estimate chunks
    total_chunks = 0
    session_plans: list[tuple[TranscriptInfo, int]] = []
    for t in transcripts:
        turns = extract_full_conversation(t.path)
        total_text = sum(len(turn.text) for turn in turns)
        est_chunks = max(1, total_text // CHUNK_TARGET_CHARS)
        session_plans.append((t, est_chunks))
        total_chunks += est_chunks

        if args.dry_run:
            logging.info(
                "  %s | %s | %6.1fKB | %4d turns | %6dK chars | ~%d chunks",
                t.date, t.session_id[:8], t.size / 1024, len(turns),
                total_text // 1000, est_chunks,
            )

    est_cost = total_chunks * FLUSH_COST_ESTIMATE
    logging.info("")
    logging.info("Estimated: %d chunks, ~$%.2f flush cost", total_chunks, est_cost)

    if args.dry_run:
        logging.info("(dry run — no LLM calls made)")
        return

    # Process sessions
    all_extractions: list[Extraction] = []
    cumulative_cost = batch_state.get("total_cost", 0.0)
    chunk_num = 0

    for t, est_chunks in session_plans:
        # Cost guard
        if args.max_cost and cumulative_cost >= args.max_cost:
            logging.info("Cost limit reached ($%.2f >= $%.2f), stopping", cumulative_cost, args.max_cost)
            break

        logging.info("")
        logging.info("Processing session %s (%s, %.1fKB)...", t.session_id[:8], t.date, t.size / 1024)

        turns = extract_full_conversation(t.path)
        if not turns:
            logging.info("  No text turns found, skipping")
            continue

        tool_summary = extract_tool_summary(t.path)
        chunks = chunk_conversation(turns)

        logging.info("  %d turns, %d chunks", len(turns), len(chunks))

        for i, chunk in enumerate(chunks):
            chunk_num += 1

            if args.max_cost and cumulative_cost >= args.max_cost:
                break

            meta_parts = [f"Session date: {t.date}"]
            if len(chunks) > 1:
                meta_parts.append(f"Part {i + 1}/{len(chunks)} ({chunk.position})")
            if tool_summary:
                meta_parts.append(f"Tool activity: {tool_summary}")
            session_meta = "\n".join(meta_parts)

            logging.info(
                "  [%d/%d] Flushing chunk %d/%d (%s, %dK chars) — $%.2f spent",
                chunk_num, total_chunks, i + 1, len(chunks),
                chunk.position, chunk.char_count // 1000, cumulative_cost,
            )

            response = await flush_chunk(chunk.text, session_meta)
            cost = FLUSH_COST_ESTIMATE
            cumulative_cost += cost

            all_extractions.append(Extraction(
                session_id=t.session_id,
                date=t.date,
                time_str=t.mtime.strftime("%H:%M"),
                chunk_position=chunk.position,
                content=response,
                cost=cost,
            ))

            # Brief pause between API calls
            await asyncio.sleep(0.5)

        # Mark session as processed
        processed[t.session_id] = {
            "chunks": len(chunks),
            "cost": len(chunks) * FLUSH_COST_ESTIMATE,
            "flushed_at": datetime.now(timezone.utc).isoformat(),
        }
        batch_state["processed_sessions"] = processed
        batch_state["total_cost"] = cumulative_cost
        save_batch_state(batch_state)

    # Write daily logs
    meaningful = [e for e in all_extractions if "FLUSH_OK" not in e.content and "FLUSH_ERROR" not in e.content]
    logging.info("")
    logging.info("=== Writing Daily Logs ===")
    logging.info("Total extractions: %d (%d meaningful)", len(all_extractions), len(meaningful))

    written = write_daily_logs(all_extractions)
    logging.info("Wrote %d daily log files", len(written))
    for p in written:
        logging.info("  %s", p.name)

    logging.info("")
    logging.info("Total cost: $%.2f", cumulative_cost)

    # Optionally trigger compilation
    if args.compile:
        logging.info("")
        logging.info("=== Triggering Compilation ===")
        import subprocess
        cmd = ["uv", "run", "--directory", str(ROOT), "python", str(SCRIPTS_DIR / "compile.py"), "--all"]
        logging.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Batch extract knowledge from historical transcripts")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be processed")
    parser.add_argument("--compile", action="store_true", help="Run compile.py after extraction")
    parser.add_argument("--max-cost", type=float, default=None, help="Stop after spending this much ($)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed sessions")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated dates to process (YYYY-MM-DD)")
    parser.add_argument(
        "--transcripts-dir", type=str, default=str(DEFAULT_TRANSCRIPTS_DIR),
        help="Directory containing JSONL transcripts",
    )
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
