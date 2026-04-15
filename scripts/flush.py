"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os

os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_FILE = SCRIPTS_DIR / "last-flush.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"

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
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def append_to_daily_log(content: str, section: str = "Session") -> None:
    """Append content to today's daily log."""
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    entry = f"### {section} ({time_str})\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


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
of important items that should be preserved in the daily log.
Do NOT use any tools — just return plain text.

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

{context}"""

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


AUTO_COMPILE_STALE_AFTER_SECONDS = 24 * 60 * 60
AUTO_COMPILE_COOLDOWN_SECONDS = 15 * 60
AUTO_COMPILE_TRIGGER_KEY = "auto_compile_triggered_at"


def load_compile_state() -> dict:
    compile_state_file = SCRIPTS_DIR / "state.json"
    if compile_state_file.exists():
        try:
            return json.loads(compile_state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def has_pending_daily_log_changes(now: datetime, compile_state: dict) -> bool:
    """Return whether today's daily log differs from the last compiled hash."""
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    log_path = DAILY_DIR / today_log
    if not log_path.exists():
        return False

    ingested = compile_state.get("ingested", {})
    if today_log in ingested:
        from hashlib import sha256

        try:
            current_hash = sha256(log_path.read_bytes()).hexdigest()[:16]
        except OSError:
            return True
        if ingested[today_log].get("hash") == current_hash:
            return False

    return True


def get_last_successful_compile_at(compile_state: dict) -> datetime | None:
    """Return the most recent successful compile timestamp from state.json."""
    latest_compile = None

    for entry in compile_state.get("ingested", {}).values():
        compiled_at = entry.get("compiled_at")
        if not compiled_at:
            continue

        try:
            compile_time = datetime.fromisoformat(compiled_at)
        except ValueError:
            continue

        if compile_time.tzinfo is None:
            compile_time = compile_time.replace(tzinfo=timezone.utc)
        else:
            compile_time = compile_time.astimezone(timezone.utc)

        if latest_compile is None or compile_time > latest_compile:
            latest_compile = compile_time

    return latest_compile


def maybe_trigger_compilation(flush_state: dict | None = None) -> None:
    """Trigger background compilation when pending changes are stale enough."""
    import subprocess as _sp

    now_utc = datetime.now(timezone.utc)
    now = now_utc.astimezone()
    compile_state = load_compile_state()

    if not has_pending_daily_log_changes(now, compile_state):
        return

    last_compile_at = get_last_successful_compile_at(compile_state)
    if last_compile_at is not None:
        compile_age_seconds = (now_utc - last_compile_at).total_seconds()
        if compile_age_seconds < AUTO_COMPILE_STALE_AFTER_SECONDS:
            return

    state = flush_state if flush_state is not None else load_flush_state()
    try:
        last_triggered_at = float(state.get(AUTO_COMPILE_TRIGGER_KEY, 0))
    except (TypeError, ValueError):
        last_triggered_at = 0.0

    if time.time() - last_triggered_at < AUTO_COMPILE_COOLDOWN_SECONDS:
        return

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    state[AUTO_COMPILE_TRIGGER_KEY] = time.time()
    save_flush_state(state)

    if last_compile_at is None:
        logging.info(
            "Auto-compilation triggered for pending changes with no prior successful compile"
        )
    else:
        logging.info(
            "Auto-compilation triggered for stale pending changes; last successful compile was %.1f hours ago",
            (now_utc - last_compile_at).total_seconds() / 3600,
        )

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


def main():
    if len(sys.argv) < 3:
        logging.error("Usage: %s <context_file.md> <session_id>", sys.argv[0])
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]

    logging.info("flush.py started for session %s, context: %s", session_id, context_file)

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds
    state = load_flush_state()
    if state.get("session_id") == session_id and time.time() - state.get("timestamp", 0) < 60:
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Read pre-extracted context
    context = context_file.read_text(encoding="utf-8").strip()
    if not context:
        logging.info("Context file is empty, skipping")
        context_file.unlink(missing_ok=True)
        return

    logging.info("Flushing session %s: %d chars", session_id, len(context))

    # Run the LLM extraction
    response = asyncio.run(run_flush(context))

    # Append to daily log
    if "FLUSH_OK" in response:
        logging.info("Result: FLUSH_OK")
        append_to_daily_log("FLUSH_OK - Nothing worth saving from this session", "Memory Flush")
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        append_to_daily_log(response, "Memory Flush")
    else:
        logging.info("Result: saved to daily log (%d chars)", len(response))
        append_to_daily_log(response, "Session")

    # Update dedup state
    state["session_id"] = session_id
    state["timestamp"] = time.time()
    save_flush_state(state)

    # Clean up context file
    context_file.unlink(missing_ok=True)

    # Trigger background compilation when today's log changed and the last
    # successful compile is stale enough to warrant a refresh.
    maybe_trigger_compilation(state)

    logging.info("Flush complete for session %s", session_id)


if __name__ == "__main__":
    main()
