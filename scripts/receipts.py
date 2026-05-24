"""Privacy-safe receipt helpers for memory capture events.

Receipts are opt-in and intentionally avoid raw transcript text, file paths,
Claude prompts, memory summaries, and knowledge-base article content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_RECEIPTS_FILE = SCRIPTS_DIR / "memory-receipts.jsonl"

_TRUTHY = {"1", "true", "yes", "on"}


def receipts_enabled() -> bool:
    """Return True when memory receipts are explicitly enabled."""
    return os.environ.get("CMC_RECEIPTS", "").strip().lower() in _TRUTHY


def _receipt_file() -> Path:
    configured = os.environ.get("CMC_RECEIPTS_FILE", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_RECEIPTS_FILE


def _hmac_identifier(value: str | None) -> str | None:
    """Hash identifiers only when a local HMAC key is configured.

    Session ids and other runtime identifiers can be guessable. If the operator
    has not configured a key, receipts still report counts/outcomes but omit
    identifier hashes instead of falling back to dictionary-attackable SHA-256.
    """
    if not value:
        return None

    key = os.environ.get("CMC_RECEIPT_HMAC_KEY", "")
    if not key:
        return None

    digest = hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:24]


def build_memory_receipt(event_name: str, *, session_id: str | None = None, **fields: Any) -> dict:
    """Build a redacted receipt for a memory pipeline event."""
    session_id_hash = _hmac_identifier(session_id)

    receipt: dict[str, Any] = {
        "schema": "claude_memory_compiler.receipt.v1",
        "event": event_name,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "session_id_hash": session_id_hash,
        "privacy": {
            "raw_context_included": False,
            "raw_transcript_path_included": False,
            "raw_memory_included": False,
            "raw_prompt_included": False,
            "hash_algorithm": "hmac-sha256:24" if session_id_hash else None,
            "hashes_omitted_reason": None if session_id_hash else "CMC_RECEIPT_HMAC_KEY not set",
        },
    }
    receipt.update(fields)
    return receipt


def write_memory_receipt(event_name: str, *, session_id: str | None = None, **fields: Any) -> None:
    """Append an opt-in receipt JSON object to the receipts JSONL file."""
    if not receipts_enabled():
        return

    receipt = build_memory_receipt(event_name, session_id=session_id, **fields)
    path = _receipt_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
