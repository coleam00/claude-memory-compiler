"""Unit tests for the hook -> flush.py -> Capture Record cutover.

Exercises hook argv propagation (--source-cwd), direct Capture Record result
statuses, PreCompact/SessionEnd identity stability, and the absence of any
daily-log append — all against temp paths with the Agent SDK, subprocesses,
and capture_records stubbed out. No network access.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
HOOKS_DIR = REPO_ROOT / "hooks"

# flush.py imports the real config module (DEFAULT_MODEL) from scripts/.
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Stub capture_records BEFORE importing any sibling module that imports it,
# so tests never depend on the real module being present or complete.
# ---------------------------------------------------------------------------



def _stub_write_capture_record(content: str, **kwargs):
    return Path(str(kwargs.get("record_day") or "unknown")) / "record.md"


_stub = types.ModuleType("capture_records")
_stub.write_capture_record = _stub_write_capture_record
_stub.list_capture_records = lambda: ()
sys.modules.setdefault("capture_records", _stub)


def _load_module(unique_name: str, path: Path):
    """Load a repo module under a unique name, with the recursion guard off.

    logging.FileHandler is stubbed so basicConfig() never creates or appends
    to the repo's scripts/*.log files during tests.
    """
    saved = os.environ.pop("CLAUDE_INVOKED_BY", None)
    try:
        spec = importlib.util.spec_from_file_location(unique_name, path)
        mod = importlib.util.module_from_spec(spec)
        # Required before exec: dataclass processing resolves the module's own
        # globals via sys.modules[cls.__module__].
        sys.modules[unique_name] = mod
        with mock.patch("logging.FileHandler", lambda *a, **k: logging.NullHandler()):
            spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is not None:
            os.environ["CLAUDE_INVOKED_BY"] = saved


def _make_transcript(path: Path, turns: int) -> Path:
    lines = [
        json.dumps(
            {
                "timestamp": "2026-08-24T10:00:00Z",
                "message": {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"},
            }
        )
        for i in range(turns)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_hook(hook_filename: str, payload: dict, unique_name: str):
    """Load a hook module fresh and run its main() with stdin faked.

    Popen is stubbed (no subprocesses) and STATE_DIR/SCRIPTS_DIR point at a
    temp dir so hook context files and stderr logs never touch the repo.
    """
    mod = _load_module(unique_name, HOOKS_DIR / hook_filename)
    popen = mock.Mock()
    stdin = mock.MagicMock()
    stdin.read.return_value = json.dumps(payload)
    with (
        tempfile.TemporaryDirectory() as td, mock.patch.object(mod, "STATE_DIR", Path(td)),
        mock.patch.object(mod, "SCRIPTS_DIR", Path(td)),
        mock.patch.object(mod.subprocess, "Popen", popen),
        mock.patch.object(sys, "stdin", stdin),
    ):
        mod.main()
    return mod, popen


# ---------------------------------------------------------------------------
# Hook argv propagation
# ---------------------------------------------------------------------------


class HookArgvPropagationTests(unittest.TestCase):
    SESSION_ID = "abc123-session"
    SOURCE_CWD = "/Users/prime/Working/some-project"

    def _popen_cmd(self, popen) -> list[str]:
        import subprocess

        args, kwargs = popen.call_args
        cmd = args[0]
        # stdout stays on DEVNULL; stderr must remain a real writable handle
        # (the append-mode flush-stderr log); no detach flags off Windows.
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIsNot(kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(hasattr(kwargs["stderr"], "write"))
        if sys.platform != "win32":
            self.assertEqual(kwargs.get("creationflags", 0), 0)
        return cmd

    def test_session_end_passes_source_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            transcript = _make_transcript(td / "transcript.jsonl", turns=4)
            payload = {
                "session_id": self.SESSION_ID,
                "source": "exit",
                "cwd": self.SOURCE_CWD,
                "transcript_path": str(transcript),
            }
            _, popen = _run_hook("session-end.py", payload, "hooktest_session_end")
            cmd = self._popen_cmd(popen)

        self.assertIn("--source-cwd", cmd)
        self.assertEqual(cmd[cmd.index("--source-cwd") + 1], self.SOURCE_CWD)
        self.assertIn(self.SESSION_ID, cmd)
        # Context file argument comes right before the session id.
        sid_index = cmd.index(self.SESSION_ID)
        self.assertTrue(cmd[sid_index - 1].endswith(".md"))

    def test_pre_compact_passes_source_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            transcript = _make_transcript(td / "transcript.jsonl", turns=6)
            payload = {
                "session_id": self.SESSION_ID,
                "cwd": self.SOURCE_CWD,
                "transcript_path": str(transcript),
            }
            _, popen = _run_hook("pre-compact.py", payload, "hooktest_pre_compact")
            cmd = self._popen_cmd(popen)

        self.assertIn("--source-cwd", cmd)
        self.assertEqual(cmd[cmd.index("--source-cwd") + 1], self.SOURCE_CWD)
        self.assertIn(self.SESSION_ID, cmd)


# ---------------------------------------------------------------------------
# flush.py direct Capture Record results
# ---------------------------------------------------------------------------


class FlushResultBase(unittest.TestCase):
    SESSION_ID = "sess-42"
    SOURCE_CWD = "/Users/prime/Working/obsidian"

    def setUp(self):
        self.writes = mock.Mock()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.td = Path(self._tmp.name)
        self.flush = _load_module("flushtest_flush", SCRIPTS_DIR / "flush.py")
        self.state_file = self.td / "last-flush.json"
        self.context_file = self.td / "context.md"

    def run_flush_main(self, response: str, context: str = "conversation context\n") -> int:
        """Drive flush.main() end-to-end with SDK + compile trigger stubbed."""
        self.context_file.write_text(context, encoding="utf-8")

        async def fake_run_flush(context: str) -> str:
            return response

        with (
            mock.patch.object(self.flush, "STATE_FILE", self.state_file),
            mock.patch.object(self.flush, "write_capture_record", self.writes),
            mock.patch.object(self.flush, "maybe_trigger_compilation"),
            mock.patch.object(self.flush, "run_flush", fake_run_flush),
            mock.patch.object(
                sys, "argv",
                [
                    "flush.py", str(self.context_file), self.SESSION_ID,
                    "--source-cwd", self.SOURCE_CWD,
                ],
            ),
        ):
            return self.flush.main()

    def single_write_kwargs(self) -> dict:
        """Assert exactly one Capture Record write happened and return kwargs."""
        self.assertEqual(self.writes.call_count, 1)
        return self.writes.call_args.kwargs


class FlushDirectResultStatusTests(FlushResultBase):
    def test_narrative_result_writes_narrative_status(self):
        rc = self.run_flush_main("**Context:** worked on things\n\n**Decisions Made:**\n- pick X")
        self.assertEqual(rc, 0)
        kwargs = self.single_write_kwargs()
        self.assertEqual(kwargs["capture_kind"], "memory_flush")
        self.assertEqual(kwargs["capture_status"], "narrative")
        self.assertEqual(kwargs["session_id"], self.SESSION_ID)
        self.assertEqual(kwargs["source_cwd"], self.SOURCE_CWD)
        self.assertEqual(kwargs["record_day"], datetime.now(UTC).astimezone().strftime("%Y-%m-%d"))
        self.assertIsInstance(kwargs["captured_at"], datetime)

    def test_no_op_result_writes_no_op_status(self):
        rc = self.run_flush_main("FLUSH_OK")
        self.assertEqual(rc, 0)
        kwargs = self.single_write_kwargs()
        self.assertEqual(kwargs["capture_status"], "no_op")

    def test_failure_result_writes_failure_status_and_nonzero_rc(self):
        rc = self.run_flush_main("FLUSH_ERROR: sdk exploded")
        self.assertEqual(rc, 1)
        kwargs = self.single_write_kwargs()
        self.assertEqual(kwargs["capture_status"], "failure")


class FlushIdentityAndDailyGuardTests(FlushResultBase):
    def test_overlap_resolves_to_single_direct_identity(self):
        """PreCompact then SessionEnd for the same (day, cwd, session) must
        produce identical identity inputs so capture_records' atomic overwrite
        collapses them into one Capture Record."""
        self.run_flush_main("first pass narrative content", context="first pass context\n")
        # Second firing overwrites the context file (fresh, richer extraction),
        # so the 60-second duplicate guard must not swallow it.
        self.run_flush_main("second pass narrative content", context="second pass richer context\n")

        self.assertEqual(self.writes.call_count, 2)
        calls = self.writes.call_args_list
        identities = [
            (c.kwargs["record_day"], c.kwargs["source_cwd"], c.kwargs["session_id"])
            for c in calls
        ]
        self.assertEqual(identities[0], identities[1])
        # Content differed, so this was an overwrite of one record, not a dup skip.
        self.assertNotEqual(calls[0].args[0], calls[1].args[0])

    def test_no_daily_log_append_anywhere_in_flow(self):
        daily_dir = REPO_ROOT / "daily"

        def daily_snapshot():
            if not daily_dir.exists():
                return []
            return sorted((p.name, p.stat().st_size) for p in daily_dir.glob("*.md"))

        before = daily_snapshot()
        self.run_flush_main("narrative content")
        self.run_flush_main("FLUSH_OK", context="nothing worth saving here\n")
        self.run_flush_main("FLUSH_ERROR: boom", context="boom context\n")
        self.assertEqual(daily_snapshot(), before)

    def test_sixty_second_guard_still_blocks_duplicate_launch(self):
        self.context_file.write_text("ctx\n", encoding="utf-8")
        async def fake_run_flush(context: str) -> str:
            return "narrative"

        argv = ["flush.py", str(self.context_file), self.SESSION_ID]
        with (
            mock.patch.object(self.flush, "STATE_FILE", self.state_file),
            mock.patch.object(self.flush, "write_capture_record", self.writes),
            mock.patch.object(self.flush, "maybe_trigger_compilation"),
            mock.patch.object(self.flush, "run_flush", fake_run_flush),
        ):
            self.state_file.write_text(
                json.dumps({"session_id": self.SESSION_ID, "timestamp": __import__("time").time()}),
                encoding="utf-8",
            )
            with mock.patch.object(sys, "argv", argv):
                rc = self.flush.main()
        self.assertEqual(rc, 0)
        self.assertEqual(self.writes.call_count, 0)


# ---------------------------------------------------------------------------
# backfill.py dedup + source project passthrough
# ---------------------------------------------------------------------------


class BackfillDedupTests(unittest.TestCase):
    DAY = "2026-08-24"
    PROJECT_DIR_NAME = "-Users-prime-Working-demo"
    DECODED_CWD = "/Users/prime/Working/demo"
    CAPTURED_SID = "11111111-2222-3333-4444-555555555555"
    FRESH_SID = "99999999-8888-7777-6666-555555555555"

    def setUp(self):
        self.backfill = _load_module("backfilltest_backfill", SCRIPTS_DIR / "backfill.py")

    def test_decode_project_cwd(self):
        self.assertEqual(self.backfill.decode_project_cwd(self.PROJECT_DIR_NAME), self.DECODED_CWD)
        self.assertIsNone(self.backfill.decode_project_cwd("not-a-encoded-path"))

    def test_rerun_skips_captured_sessions_and_passes_source_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proj_dir = td / "projects" / self.PROJECT_DIR_NAME
            proj_dir.mkdir(parents=True)
            for sid in (self.CAPTURED_SID, self.FRESH_SID):
                _make_transcript(proj_dir / f"{sid}.jsonl", turns=3)

            records = [
                SimpleNamespace(
                    record_day=self.DAY,
                    source_sid=f"backfill-{self.DAY}-{self.CAPTURED_SID}",
                )
            ]

            sub_run = mock.Mock()
            argv = ["backfill.py", "--date", self.DAY, "--yes", "--no-compile"]
            with (
                mock.patch.object(self.backfill, "PROJECTS_DIR", td / "projects"),
                mock.patch.object(self.backfill, "CONTEXT_DIR", td),
                mock.patch.object(self.backfill, "list_capture_records", return_value=records),
                mock.patch.object(self.backfill.subprocess, "run", sub_run),
                mock.patch.object(sys, "argv", argv),
            ):
                rc = self.backfill.main()

            self.assertEqual(rc, 0)
            self.assertEqual(sub_run.call_count, 1)
            cmd = sub_run.call_args.args[0]
            flushed_sid = f"backfill-{self.DAY}-{self.FRESH_SID}"
            self.assertIn(flushed_sid, cmd)
            self.assertNotIn(f"backfill-{self.DAY}-{self.CAPTURED_SID}", cmd)
            self.assertIn("--source-cwd", cmd)
            self.assertEqual(cmd[cmd.index("--source-cwd") + 1], self.DECODED_CWD)


# ---------------------------------------------------------------------------
# Atomic flush-state persistence
# ---------------------------------------------------------------------------


class FlushStateAtomicityTests(unittest.TestCase):
    """save_flush_state must atomically replace and clean up its temp file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.td = Path(self._tmp.name)
        self.flush = _load_module("flushtest_atomic_flush", SCRIPTS_DIR / "flush.py")
        self.state_file = self.td / "flush-state.json"

    def _leftovers(self) -> list:
        return sorted(p.name for p in self.td.iterdir() if p.name != self.state_file.name)

    def test_save_replaces_state_atomically_with_fsync(self):
        state = {"session_id": "sess-1", "timestamp": 123.5}
        fsync = mock.Mock()
        with (
            mock.patch.object(self.flush, "STATE_FILE", self.state_file),
            mock.patch.object(self.flush.os, "fsync", fsync),
        ):
            self.flush.save_flush_state(state)
        self.assertEqual(json.loads(self.state_file.read_text(encoding="utf-8")), state)
        fsync.assert_called_once()  # payload is durable before the rename
        self.assertEqual(self._leftovers(), [])  # temp file consumed by replace

    def test_replace_failure_keeps_old_state_and_cleans_temp(self):
        self.state_file.write_text('{"keep": true}', encoding="utf-8")
        with (
            mock.patch.object(self.flush, "STATE_FILE", self.state_file),
            mock.patch.object(self.flush.os, "replace", side_effect=OSError("disk full")),
        ):
            with self.assertRaises(OSError):
                self.flush.save_flush_state({"session_id": "later"})
        self.assertEqual(json.loads(self.state_file.read_text(encoding="utf-8")), {"keep": True})
        self.assertEqual(self._leftovers(), [])

    def test_write_failure_cleans_temp_file(self):
        broken = mock.MagicMock()
        broken.__enter__.return_value.write.side_effect = OSError("io error")
        with (
            mock.patch.object(self.flush, "STATE_FILE", self.state_file),
            mock.patch.object(self.flush.os, "fdopen", return_value=broken),
        ):
            with self.assertRaises(OSError):
                self.flush.save_flush_state({"session_id": "x"})
        self.assertEqual([p.name for p in self.td.iterdir()], [])


# ---------------------------------------------------------------------------
# Raw Input tagging: LF/CRLF tolerance and byte preservation
# ---------------------------------------------------------------------------


class RawTagCRLFTests(unittest.TestCase):
    """Raw tagging accepts LF and CRLF, preserves bytes, never duplicates YAML."""

    RAW_TAG = "memory-compiler/raw-input"
    CRLF_RAW = (
        b"---\r\n"
        b"created: 2026-08-24T10:00:00\r\n"
        b"tags: []\r\n"
        b"source: hook\r\n"
        b"---\r\n"
        b"Body stays.\r\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.td = Path(self._tmp.name)
        # Load the REAL capture_records under a unique name: sys.modules holds
        # a stub for the plain name (see module top).
        self.cr = _load_module(
            "capture_records_real_fixture", SCRIPTS_DIR / "capture_records.py"
        )

    def apply(self, data: bytes, *, compiled_in=None) -> bytes:
        return self.cr._apply_raw_tags(data, [self.RAW_TAG], compiled_in)

    def test_crlf_frontmatter_updated_in_place_never_duplicated(self):
        out = self.apply(self.CRLF_RAW)
        self.assertEqual(out.count(b"---"), 2)  # exactly one YAML block
        self.assertTrue(out.startswith(b"---\r\n"))
        self.assertIn(b'tags: ["memory-compiler/raw-input"]', out)

    def test_crlf_newline_style_and_body_preserved(self):
        out = self.apply(self.CRLF_RAW)
        self.assertNotIn(b"\n", out.replace(b"\r\n", b""))  # every EOL is CRLF
        self.assertIn(b"source: hook\r\n", out)
        self.assertTrue(out.endswith(b"Body stays.\r\n"))

    def test_crlf_tagging_is_idempotent(self):
        once = self.apply(self.CRLF_RAW)
        self.assertEqual(self.apply(once), once)

    def test_lf_merge_and_compiled_in_unchanged(self):
        lf = b'---\ncreated: t\ntags: ["old"]\n---\nBody.\n'
        out = self.apply(lf, compiled_in="cap_9")
        self.assertEqual(out.count(b"---"), 2)
        self.assertIn(b'tags: ["old", "memory-compiler/raw-input"]', out)
        self.assertIn(b'compiled_in: "cap_9"', out)
        self.assertTrue(out.endswith(b"Body.\n"))

    def test_missing_frontmatter_prepends_exactly_one_block(self):
        body = b"plain notes\r\nsecond line\r\n"
        out = self.apply(body)
        self.assertEqual(out.count(b"---"), 2)
        self.assertTrue(out.endswith(body))  # body bytes untouched

    def test_tag_raw_inputs_round_trip_on_crlf_file(self):
        path = self.td / "raw.md"
        path.write_bytes(self.CRLF_RAW)
        lock = self.td / ".lock"
        with mock.patch.object(self.cr, "CAPTURE_LOCK_FILE", lock):
            self.cr.tag_raw_inputs((path,), compiled=True, capture_id="cap_7")
        data = path.read_bytes()
        self.assertEqual(data.count(b"---"), 2)
        self.assertIn(b'compiled_in: "cap_7"', data)
        self.assertTrue(data.endswith(b"Body stays.\r\n"))
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        before = data
        with mock.patch.object(self.cr, "CAPTURE_LOCK_FILE", lock):
            self.cr.tag_raw_inputs((path,), compiled=True, capture_id="cap_7")
        self.assertEqual(path.read_bytes(), before)  # re-tag is a byte-level no-op
        self.assertFalse(any(p.suffix == ".tmp" for p in self.td.iterdir()))


if __name__ == "__main__":
    unittest.main()
