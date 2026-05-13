"""Smoke tests for state.json resilience in utils.py.

Verifies that:
- load_state() returns the default state when state.json is missing.
- load_state() returns the default state when state.json is empty
  (the bug that stalled auto-compile for ~30 days in the wild — a
  truncated write from a crashed save left a zero-byte file, and every
  subsequent compile crashed in json.loads(b'')).
- load_state() returns the default state when state.json is malformed,
  preserving the bad file as state.json.broken for inspection.
- save_state() is atomic: an OSError between the temp-write and the
  os.replace must leave the previous good state.json intact, not a
  truncated file.

Run from repo root:
    uv run python scripts/test_utils_state.py
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class StateResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.scripts_dir = Path(self._tmpdir.name)
        self.state_file = self.scripts_dir / "state.json"
        # Reroute config.STATE_FILE for the duration of each test.
        # config.py is imported at utils.py import time, so we patch
        # the already-imported attribute.
        import config  # type: ignore[import-not-found]
        self._orig_state_file = config.STATE_FILE
        config.STATE_FILE = self.state_file
        # utils.py imported STATE_FILE by name — re-bind that too.
        import utils  # type: ignore[import-not-found]
        self._orig_utils_state_file = utils.STATE_FILE
        utils.STATE_FILE = self.state_file
        self.utils = utils
        self.config = config

    def tearDown(self) -> None:
        self.config.STATE_FILE = self._orig_state_file
        self.utils.STATE_FILE = self._orig_utils_state_file
        self._tmpdir.cleanup()

    def test_load_missing_returns_default(self) -> None:
        state = self.utils.load_state()
        self.assertEqual(state, {"ingested": {}, "query_count": 0,
                                 "last_lint": None, "total_cost": 0.0})
        # Default must be a fresh dict — mutating it must not leak.
        state["ingested"]["x"] = 1
        again = self.utils.load_state()
        self.assertEqual(again["ingested"], {})

    def test_load_empty_file_returns_default(self) -> None:
        self.state_file.write_text("", encoding="utf-8")
        state = self.utils.load_state()
        self.assertEqual(state["ingested"], {})

    def test_load_whitespace_only_returns_default(self) -> None:
        self.state_file.write_text("   \n\t  \n", encoding="utf-8")
        state = self.utils.load_state()
        self.assertEqual(state["ingested"], {})

    def test_load_malformed_quarantines_and_returns_default(self) -> None:
        self.state_file.write_text("{not json", encoding="utf-8")
        state = self.utils.load_state()
        self.assertEqual(state["ingested"], {})
        # Bad file should have been moved aside.
        broken = self.state_file.with_suffix(".json.broken")
        self.assertTrue(broken.exists(), "broken file should be preserved")
        self.assertFalse(self.state_file.exists(),
                         "state.json should be cleared after quarantine")

    def test_load_valid_state_round_trip(self) -> None:
        payload = {"ingested": {"2026-04-12.md": {"hash": "abc", "cost_usd": 0.5}},
                   "query_count": 3, "last_lint": "2026-05-12", "total_cost": 0.5}
        self.state_file.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.utils.load_state(), payload)

    def test_save_is_atomic_on_crash(self) -> None:
        good = {"ingested": {"a": 1}, "query_count": 0,
                "last_lint": None, "total_cost": 0.0}
        self.utils.save_state(good)
        # Confirm baseline.
        self.assertEqual(json.loads(self.state_file.read_text()), good)

        # Simulate a crash between temp-write and os.replace.
        with patch("utils.os.replace", side_effect=OSError("simulated crash")):
            try:
                self.utils.save_state({"ingested": {"b": 2}})
            except OSError:
                pass

        # The previous good file must still be intact.
        loaded = json.loads(self.state_file.read_text())
        self.assertEqual(loaded, good,
                         "atomic save must leave previous state intact on crash")

        # And no .tmp orphan should be readable as state.
        tmp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        if tmp.exists():
            # Clean up artifact but make sure it doesn't shadow.
            tmp.unlink()

    def test_save_then_load_round_trip(self) -> None:
        payload = {"ingested": {"x": {"hash": "h"}}, "query_count": 7,
                   "last_lint": None, "total_cost": 1.23}
        self.utils.save_state(payload)
        self.assertEqual(self.utils.load_state(), payload)


if __name__ == "__main__":
    # Ensure scripts/ is on sys.path so 'import config' / 'import utils' work
    # when run as `uv run python scripts/test_utils_state.py` from repo root.
    import sys
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    unittest.main()
