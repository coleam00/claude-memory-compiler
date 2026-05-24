from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.receipts import build_memory_receipt, write_memory_receipt


class MemoryReceiptTests(unittest.TestCase):
    def test_missing_hmac_key_omits_guessable_session_hash(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            receipt = build_memory_receipt(
                "memory.flush.requested",
                session_id="session-acme-secret",
                trigger="SessionEnd",
                context_chars=123,
            )

        serialized = json.dumps(receipt)
        self.assertIsNone(receipt["session_id_hash"])
        self.assertIn("CMC_RECEIPT_HMAC_KEY not set", serialized)
        self.assertNotIn("session-acme-secret", serialized)

    def test_hmac_key_hashes_identifier_without_raw_value(self) -> None:
        with patch.dict(os.environ, {"CMC_RECEIPT_HMAC_KEY": "local-test-key"}, clear=True):
            receipt = build_memory_receipt(
                "memory.flush.completed",
                session_id="session-acme-secret",
                outcome="saved",
                summary_chars=456,
            )

        serialized = json.dumps(receipt)
        self.assertRegex(receipt["session_id_hash"], r"^[0-9a-f]{24}$")
        self.assertNotIn("session-acme-secret", serialized)
        self.assertEqual(receipt["privacy"]["hash_algorithm"], "hmac-sha256:24")

    def test_write_receipt_is_opt_in_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt_file = Path(tmpdir) / "receipts.jsonl"
            env = {
                "CMC_RECEIPTS": "1",
                "CMC_RECEIPTS_FILE": str(receipt_file),
                "CMC_RECEIPT_HMAC_KEY": "local-test-key",
            }
            with patch.dict(os.environ, env, clear=True):
                write_memory_receipt(
                    "memory.flush.requested",
                    session_id="session-acme-secret",
                    trigger="SessionEnd",
                    turn_count=3,
                    context_chars=789,
                )

            payload = receipt_file.read_text(encoding="utf-8")
            self.assertIn("memory.flush.requested", payload)
            self.assertIn('"raw_context_included": false', payload)
            self.assertNotIn("session-acme-secret", payload)


if __name__ == "__main__":
    unittest.main()
