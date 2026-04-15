from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import flush


class FlushStateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.scripts_dir = self.root / "scripts"
        self.daily_dir = self.root / "daily"
        self.scripts_dir.mkdir()
        self.daily_dir.mkdir()

        self.state_file = self.scripts_dir / "last-flush.json"
        self.lock_file = self.scripts_dir / "last-flush.lock"
        self.compile_script = self.scripts_dir / "compile.py"
        self.compile_script.write_text("print('ok')\n", encoding="utf-8")

        self.patchers = [
            mock.patch.object(flush, "ROOT", self.root),
            mock.patch.object(flush, "SCRIPTS_DIR", self.scripts_dir),
            mock.patch.object(flush, "DAILY_DIR", self.daily_dir),
            mock.patch.object(flush, "STATE_FILE", self.state_file),
            mock.patch.object(flush, "FLUSH_STATE_LOCK_FILE", self.lock_file),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def read_state(self) -> dict:
        if not self.state_file.exists():
            return {}
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def write_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def test_record_flush_state_preserves_existing_cooldown(self) -> None:
        self.write_state({flush.AUTO_COMPILE_TRIGGER_KEY: 123.0})

        flush.record_flush_state("session-1", 456.0)

        self.assertEqual(
            self.read_state(),
            {
                flush.AUTO_COMPILE_TRIGGER_KEY: 123.0,
                "session_id": "session-1",
                "timestamp": 456.0,
            },
        )
        self.assertFalse(self.lock_file.exists())

    def test_maybe_trigger_compilation_only_spawns_once_under_concurrency(self) -> None:
        spawn_count = 0
        spawn_count_lock = threading.Lock()
        start_barrier = threading.Barrier(3)

        def fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal spawn_count
            with spawn_count_lock:
                spawn_count += 1
            time.sleep(0.1)
            return mock.Mock()

        def worker() -> None:
            start_barrier.wait()
            flush.maybe_trigger_compilation("session-1", 10.0)

        with (
            mock.patch.object(flush, "load_compile_state", return_value={}),
            mock.patch.object(flush, "has_pending_daily_log_changes", return_value=True),
            mock.patch.object(flush, "get_last_successful_compile_at", return_value=None),
            mock.patch("subprocess.Popen", side_effect=fake_popen),
        ):
            threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
            for thread in threads:
                thread.start()
            start_barrier.wait()
            for thread in threads:
                thread.join()

        self.assertEqual(spawn_count, 1)
        self.assertIn(flush.AUTO_COMPILE_TRIGGER_KEY, self.read_state())

    def test_failed_spawn_does_not_arm_cooldown(self) -> None:
        original_state = {"session_id": "existing", "timestamp": 1.0}
        self.write_state(original_state)

        with (
            mock.patch.object(flush, "load_compile_state", return_value={}),
            mock.patch.object(flush, "has_pending_daily_log_changes", return_value=True),
            mock.patch.object(flush, "get_last_successful_compile_at", return_value=None),
            mock.patch("subprocess.Popen", side_effect=OSError("spawn failed")),
        ):
            flush.maybe_trigger_compilation("session-2", 20.0)

        self.assertEqual(self.read_state(), original_state)


if __name__ == "__main__":
    unittest.main()
