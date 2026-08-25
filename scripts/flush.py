"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and writes one direct Capture Record per result.

Usage:
    uv run python flush.py <context_file.md> <session_id> [--source-cwd <dir>]
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os

os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from shutil import which

from capture_records import write_capture_record
from config import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
LOG_FILE = SCRIPTS_DIR / "flush.log"
STATE_FILE = SCRIPTS_DIR / "flush-state.json"


def claude_cli_path() -> str:
    """Return the logged-in Claude Code CLI instead of the SDK's bundled copy."""
    path = os.environ.get("CMC_CLAUDE_CLI") or which("claude")
    if not path:
        raise RuntimeError("Claude Code CLI not found; install it or set CMC_CLAUDE_CLI")
    return path


def report_claude_stderr(line: str) -> None:
    """Forward Claude Code subprocess diagnostics to the flush log."""
    logging.error("Claude stderr: %s", line)



# Set up file-based logging so we can verify the background process ran.
# The parent process sends stdout/stderr to DEVNULL (to avoid the inherited
# file handle bug on Windows), so this is our only observability channel.
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_flush_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: dict) -> None:
    """Atomically replace the flush state via a same-directory temp file."""
    payload = json.dumps(state).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(
        dir=STATE_FILE.parent, prefix=f".{STATE_FILE.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, STATE_FILE)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

async def run_flush(context: str) -> str:
    """Use Claude Agent SDK to extract important knowledge from conversation context."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the Capture Record.
Do NOT use any tools. Return plain text only.

Format your response as a structured Capture Record with these sections:

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

{context}"""

    response = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT),
                allowed_tools=[],
                max_turns=2,
                model=os.environ.get("CMC_MODEL", DEFAULT_MODEL),
                cli_path=claude_cli_path(),
                stderr=report_claude_stderr,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage) and message.is_error:
                response = f"FLUSH_ERROR: {message.errors or message.result}"
    except Exception as e:
        import traceback
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response


COMPILE_AFTER_HOUR = 18  # 6 PM local time


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour, spawn compile.py in the background.

    compile.py owns its own dedup (payload hashes in state.json) and its
    global compile lock, so no duplicate-record hash checking happens here.
    """
    import subprocess as _sp

    now = datetime.now(UTC).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        log_handle = open(str(SCRIPTS_DIR / "compile.log"), "a")
        _sp.Popen(cmd, stdout=log_handle, stderr=_sp.STDOUT, cwd=str(ROOT), **kwargs)
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flush conversation context into a Capture Record.")
    parser.add_argument("context_file", help="Path to extracted conversation context (.md)")
    parser.add_argument("session_id", help="Source session id (used for dedup)")
    parser.add_argument("--date", default=None, help="Override record day (YYYY-MM-DD); used for backfill")
    parser.add_argument("--no-compile", action="store_true", help="Skip end-of-day compile trigger")
    parser.add_argument(
        "--source-cwd",
        default=None,
        help="Working directory of the captured session; part of direct identity",
    )
    args = parser.parse_args()

    context_file = Path(args.context_file)
    session_id = args.session_id
    date_override = args.date
    source_cwd = args.source_cwd or None
    logging.info("flush.py started for session %s, context: %s", session_id, context_file)

    # Read pre-extracted context
    context = context_file.read_text(encoding="utf-8").strip()
    if not context:
        logging.info("Context file is empty, skipping")
        context_file.unlink(missing_ok=True)
        return 0

    # Deduplication: skip only if the same session was flushed with the SAME
    # context within the last 60 seconds (duplicate hook launches of one
    # extraction). A fresh, richer extraction has different content and is
    # always processed; identity collapse happens at capture_records.
    state = load_flush_state()
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    if (
        state.get("session_id") == session_id
        and time.time() - state.get("timestamp", 0) < 60
        and state.get("context_sha256") in (None, context_sha256)
    ):
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return 0


    logging.info("Flushing session %s: %d chars", session_id, len(context))

    # Run the LLM extraction
    response = asyncio.run(run_flush(context))
    # Write one direct Capture Record. Identity is (record_day, source_cwd,
    # session_id), so PreCompact/SessionEnd overlap resolves to one record.
    now = datetime.now(UTC).astimezone()
    record_day = date_override or now.strftime("%Y-%m-%d")
    if "FLUSH_OK" in response:
        logging.info("Result: FLUSH_OK")
        write_capture_record(
            "FLUSH_OK - Nothing worth saving from this session",
            capture_kind="memory_flush",
            capture_status="no_op",
            session_id=session_id,
            source_cwd=source_cwd,
            captured_at=now,
            record_day=record_day,
        )
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        write_capture_record(
            response,
            capture_kind="memory_flush",
            capture_status="failure",
            session_id=session_id,
            source_cwd=source_cwd,
            captured_at=now,
            record_day=record_day,
        )
        return 1
    else:
        logging.info("Result: saved to Capture Record (%d chars)", len(response))
        write_capture_record(
            response,
            capture_kind="memory_flush",
            capture_status="narrative",
            session_id=session_id,
            source_cwd=source_cwd,
            captured_at=now,
            record_day=record_day,
        )
    # Save flush state only after a narrative or no-op record lands; the
    # failure branch returns earlier, so a failed flush can retry.
    save_flush_state(
        {"session_id": session_id, "timestamp": time.time(), "context_sha256": context_sha256}
    )
    if not args.no_compile:
        maybe_trigger_compilation()
    return 0

if __name__ == "__main__":
    sys.exit(main())
