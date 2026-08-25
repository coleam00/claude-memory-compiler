"""Path constants and configuration for the personal knowledge base."""

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT_DIR / "daily"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
CONCEPTS_DIR = KNOWLEDGE_DIR / "concepts"
CONNECTIONS_DIR = KNOWLEDGE_DIR / "connections"
QA_DIR = KNOWLEDGE_DIR / "qa"
REPORTS_DIR = ROOT_DIR / "reports"
SCRIPTS_DIR = ROOT_DIR / "scripts"
HOOKS_DIR = ROOT_DIR / "hooks"
AGENTS_FILE = ROOT_DIR / "AGENTS.md"

INDEX_FILE = KNOWLEDGE_DIR / "index.md"
LOG_FILE = KNOWLEDGE_DIR / "log.md"
STATE_FILE = SCRIPTS_DIR / "state.json"

# ── Records ───────────────────────────────────────────────────────────
# Direct Capture Records are written by hooks (one per logical session);
# historical daily aggregates stay readable but never receive new writes.
RECORDS_DIR = ROOT_DIR / "records"
CAPTURE_RECORDS_DIR = RECORDS_DIR / "captures"
RUNS_DIR = RECORDS_DIR / "runs"

# Markdown Raw Inputs live at the vault root, one level above this repo.
RAW_DIR = ROOT_DIR.parent / "raw"

# ── Compilation models ────────────────────────────────────────────────
# Exact OMP model identifiers; round-robin assignment keeps model counts
# balanced across pending Capture Records.
COMPILE_MODEL_PRIMARY = "anthropic/claude-sonnet-5"
COMPILE_MODEL_SECONDARY = "openai-codex/gpt-5.6-terra"
COMPILE_MODELS = (COMPILE_MODEL_PRIMARY, COMPILE_MODEL_SECONDARY)

# ── Interprocess locks ────────────────────────────────────────────────
CAPTURE_LOCK_FILE = SCRIPTS_DIR / ".capture-records.lock"
COMPILE_LOCK_FILE = SCRIPTS_DIR / ".compile.lock"

# Claude Code accepts this canonical model identifier; use it instead of the
# CLI's provider-selected default, which may be the more expensive Fable model.
DEFAULT_MODEL = "claude-sonnet-5"

# ── Timezone ───────────────────────────────────────────────────────────
TIMEZONE = "America/Chicago"
LOCAL_TZ = ZoneInfo(TIMEZONE)

def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
