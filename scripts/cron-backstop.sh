#!/usr/bin/env bash
# Memory compiler cron backstop.
#
# Runs nightly to catch any Claude Code sessions whose SessionEnd hook didn't
# fire (e.g. crashed shell, kill -9, hook regression). backfill.py is idempotent
# — it skips sessions already recorded in the target daily log via [sid:...]
# markers, so re-runs are safe and free.
#
# Targets YESTERDAY (a fully-closed day) to avoid racing with live SessionEnd
# flushes happening on today.

set -euo pipefail

# Derive ROOT from this script's own location (scripts/ -> project root) so the
# backstop survives vault relocations instead of hardcoding an absolute path.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/scripts/cron-backstop.log"

# Yesterday in local time (BSD date)
YESTERDAY=$(date -v-1d +%Y-%m-%d)

{
  echo "--- $(date -Iseconds) backstop for $YESTERDAY ---"
  cd "$ROOT"
  # Use full path to uv since launchd/cron has a minimal PATH
  PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" \
    uv run python scripts/backfill.py --date "$YESTERDAY" --yes 2>&1
  echo "--- $(date -Iseconds) done ---"
} >> "$LOG" 2>&1
