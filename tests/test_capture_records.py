"""Coverage for the shared CaptureRecord contract and projection preservation.

Run with: uv run python -m unittest tests.test_capture_records
Tests are hermetic: every filesystem-touching test redirects the module
constants into a temporary directory; repository data is never mutated.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import capture_records as cr
import materialize_records as mr

CWD = "/Users/prime/work/some-project"
T0 = datetime(2026, 8, 25, 9, 30, 0, tzinfo=UTC)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CaptureRecordsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.captures = root / "captures"
        self.raw = root / "raw"
        self.captures.mkdir()
        self.raw.mkdir()
        self._old = (cr.CAPTURE_RECORDS_DIR, cr.RAW_DIR)
        cr.CAPTURE_RECORDS_DIR = self.captures
        cr.RAW_DIR = self.raw

    def tearDown(self) -> None:
        cr.CAPTURE_RECORDS_DIR, cr.RAW_DIR = self._old
        self._tmp.cleanup()

    def write(self, content: str, *, sid: str | None = "sess-1", when=T0):
        return cr.write_capture_record(
            content,
            capture_kind="session",
            capture_status="narrative",
            session_id=sid,
            source_cwd=CWD,
            captured_at=when,
        )

    def test_stable_direct_identity_across_hook_events(self) -> None:
        first = self.write("pre-compact extract", when=T0)
        second = self.write(
            "session-end extract", sid="sess-1", when=datetime(2026, 8, 25, 18, 0, tzinfo=UTC)
        )
        self.assertEqual(first, second)
        one, two = cr.read_capture(first), cr.read_capture(second)
        self.assertEqual(one.record_id, two.record_id)
        # The latest payload wins and stays readable through the same path.
        self.assertEqual(two.payload_sha256, sha(b"session-end extract"))
        self.assertIn("session-end extract", two.body)

    def test_missing_session_id_falls_back_to_content_hash(self) -> None:
        lone = self.write("orphan payload", sid=None)
        record = cr.read_capture(lone)
        identity = hashlib.sha256(b"orphan payload").hexdigest()[:16]
        self.assertIn(identity, record.record_id)
        self.assertIn(identity, lone.name)
        self.assertIsNone(record.source_sid)
        other = self.write("different payload", sid=None)
        self.assertNotEqual(lone, other)

    def test_historical_parsing(self) -> None:
        payload_hash = sha(b"### Memory Flush (backfill 03:00) [sid:sid-abc]\n\nFLUSH_OK\n")
        historical = (
            "---\n"
            'record_type: "capture_record"\n'
            'record_id: "capture:[\\"2026-08-24\\",\\"memory_flush\\",\\"backfill 03:00\\"]"\n'
            "record_day: 2026-08-24\n"
            'capture_kind: "memory_flush"\n'
            'capture_status: "no_op"\n'
            'source_timestamp: "backfill 03:00"\n'
            'source_sid: "sid-abc"\n'
            'source_file: "daily/2026-08-24.md"\n'
            'source_aggregate_file: "daily/2026-08-24.md"\n'
            "source_ordinal: 1\n"
            f'payload_sha256: "{payload_hash}"\n'
            'source_aggregate_sha256: "deadbeef"\n'
            'tags: ["memory-compiler/capture-record"]\n'
            "---\n"
            "### Memory Flush (backfill 03:00) [sid:sid-abc]\n\nFLUSH_OK\n"
        )
        path = self.captures / "2026-08-24" / "historical.md"
        path.parent.mkdir()
        path.write_text(historical, encoding="utf-8")
        record = cr.read_capture(path)
        self.assertEqual(record.record_day, "2026-08-24")
        self.assertEqual(record.capture_status, "no_op")
        self.assertEqual(record.source_sid, "sid-abc")
        self.assertEqual(record.payload_sha256, payload_hash)
        self.assertTrue(record.body.startswith("### Memory Flush"))
        listed = cr.list_capture_records()
        self.assertEqual([r.path for r in listed], [path])
        self.assertEqual(cr.capture_state_key(listed[0]), listed[0].record_id)


class RawInputTestCase(CaptureRecordsTestCase):
    def make_raw(self, name: str, text: str) -> tuple[Path, str]:
        path = self.raw / name
        path.write_text(text, encoding="utf-8")
        return path, sha(path.read_bytes())

    RAW_A = "---\ntitle: A\ncreated: 2026-08-25\ntags: [inbox/keep]\n---\n\nBody A.\n"
    RAW_B = "---\ntitle: B\ncreated: 2026-08-24T10:00:00\n---\n\nBody B.\n"
    RAW_C = "No frontmatter here.\n"

    def test_same_day_unprocessed_discovery(self) -> None:
        a, hash_a = self.make_raw("a.md", self.RAW_A)
        self.make_raw("b.md", self.RAW_B)  # Different day.
        self.make_raw("c.md", self.RAW_C)  # No created date.
        state = {"compiled_raw_inputs": {"a.md": {"sha256": hash_a}}}
        self.assertEqual(cr.discover_unprocessed_raw("2026-08-25", state), ())
        fresh, _ = self.make_raw("d.md", self.RAW_D)
        self.assertEqual(cr.discover_unprocessed_raw("2026-08-25", state), (fresh,))
        self.assertEqual(cr.discover_unprocessed_raw("2026-08-24", {}), (self.raw / "b.md",))

    RAW_D = "---\ntitle: D\ncreated: 2026-08-25\n---\n\nBody D.\n"

    def test_tagging_preserves_yaml_and_body(self) -> None:
        path, _ = self.make_raw("keep.md", self.RAW_A)
        original_body = path.read_bytes().split(b"---\n", 2)[-1]
        cr.tag_raw_inputs((path,), compiled=False)
        text = path.read_bytes()
        self.assertIn(b"memory-compiler/raw-input", text)
        self.assertNotIn(b"memory-compiler/compiled", text)
        self.assertIn(b"title: A", text)
        self.assertIn(b"inbox/keep", text)
        self.assertTrue(text.endswith(original_body))
        cr.tag_raw_inputs((path,), compiled=True, capture_id="direct:x")
        final = path.read_bytes()
        self.assertIn(b"memory-compiler/raw-input", final)
        self.assertIn(b"memory-compiler/compiled", final)
        self.assertIn(b'compiled_in: "direct:x"', final)
        self.assertTrue(final.endswith(original_body))

    def test_tagging_creates_frontmatter_when_missing(self) -> None:
        path, _ = self.make_raw("bare.md", self.RAW_C)
        cr.tag_raw_inputs((path,), compiled=True, capture_id="direct:y")
        text = path.read_bytes()
        self.assertTrue(text.startswith(b"---\n"))
        self.assertIn(b"memory-compiler/raw-input", text)
        self.assertIn(b"compiled_in: ", text)
        self.assertTrue(text.endswith(b"No frontmatter here.\n"))


class RefreshPreservationTestCase(CaptureRecordsTestCase):
    DAILY = (
        "# 2026-08-25\n\n"
        "### Session (10:00) [sid:s-123]\n\n"
        "Narrative payload.\n\n"
    )
    LOG = "## [2026-08-25T11:00:00-07:00] compile | daily/2026-08-25.md\n\n"

    def setUp(self) -> None:
        super().setUp()
        self.daily_dir = Path(self._tmp.name) / "daily"
        self.runs = Path(self._tmp.name) / "runs"
        self.log = Path(self._tmp.name) / "log.md"
        self.daily_dir.mkdir()
        self.daily_file = self.daily_dir / "2026-08-25.md"
        self.daily_file.write_text(self.DAILY, encoding="utf-8")
        self.log.write_text(self.LOG, encoding="utf-8")
        self._old_mr = (mr.DAILY, mr.BUILD_LOG, mr.CAPTURES, mr.RUNS, mr.CAPTURE_INVARIANTS, mr.RUN_INVARIANTS)
        mr.DAILY, mr.BUILD_LOG, mr.CAPTURES, mr.RUNS = self.daily_dir, self.log, self.captures, self.runs
        mr.CAPTURE_INVARIANTS, mr.RUN_INVARIANTS = None, None
        cr.CAPTURE_RECORDS_DIR = self.captures

    def tearDown(self) -> None:
        mr.DAILY, mr.BUILD_LOG, mr.CAPTURES, mr.RUNS, mr.CAPTURE_INVARIANTS, mr.RUN_INVARIANTS = self._old_mr
        super().tearDown()

    def test_refresh_preserves_direct_records_and_refreshes_history(self) -> None:
        total, added, updated, removed = mr.refresh()
        self.assertEqual((added, removed), (2, 0))  # One capture + one run projection.
        projected = next(p for p in sorted(self.captures.rglob("*.md")) if not p.name.startswith("2026-08-25--direct"))

        direct = cr.write_capture_record(
            "live session body\n",
            capture_kind="session",
            capture_status="narrative",
            session_id="live-1",
            source_cwd=CWD,
            captured_at=T0,
        )
        direct_bytes = direct.read_bytes()

        # Aggregate grows AND shrinks: history must follow, direct must not.
        self.daily_file.write_text(self.DAILY + "### Session (11:00)\n\nMore narrative.\n\n", encoding="utf-8")
        total, added, updated, removed = mr.refresh()
        self.assertEqual(removed, 0)
        self.assertEqual(direct.read_bytes(), direct_bytes)

        self.daily_file.unlink()
        total, added, updated, removed = mr.refresh()
        self.assertEqual(removed, 2)  # Both historical capture projections went stale.
        self.assertFalse(projected.exists())
        self.assertTrue(direct.exists())
        self.assertEqual(direct.read_bytes(), direct_bytes)
        parsed = cr.read_capture(direct)
        self.assertEqual(parsed.payload_sha256, sha(b"live session body\n"))

    def _temp_residue(self) -> list[Path]:
        return [
            path
            for root in (self.captures, self.runs)
            for path in root.rglob("*")
            if path.is_file() and path.name.endswith(".tmp")
        ]

    def test_refresh_replaces_projection_atomically_without_temp_residue(self) -> None:
        total, added, updated, removed = mr.refresh()
        self.assertEqual((added, removed), (2, 0))
        projected = next(
            p
            for p in sorted(self.captures.rglob("*.md"))
            if not p.name.startswith("2026-08-25--direct")
        )
        first_bytes = projected.read_bytes()

        # Aggregate changes: the existing projection must be replaced via
        # os.replace, leaving only the final bytes behind.
        self.daily_file.write_text(self.DAILY + "### Session (12:00)\n\nLater.\n\n", encoding="utf-8")
        total, added, updated, removed = mr.refresh()
        self.assertEqual(updated, 1)
        self.assertEqual(projected.read_bytes() == first_bytes, False)
        self.assertEqual(self._temp_residue(), [])
        # Only projection notes remain: no hidden temp siblings survive.
        strays = [p.name for p in self.captures.rglob("*") if p.is_file() and p.suffix != ".md"]
        self.assertEqual(strays, [])

    def test_refresh_cleans_temp_files_when_replace_fails(self) -> None:
        with mock.patch.object(mr.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                mr.refresh()
        # No partial projection appeared and every temp file was removed.
        projected = [
            p
            for p in self.captures.rglob("*.md")
            if not p.name.startswith("2026-08-25--direct")
        ]
        self.assertEqual(projected, [])
        self.assertEqual(self._temp_residue(), [])

    def test_refresh_cleans_temp_files_when_write_fails(self) -> None:
        # os.fsync runs after the temp file exists and has been written, so a
        # failure there proves the cleanup path removes the real temp file.
        with mock.patch.object(mr.os, "fsync", side_effect=OSError("fsync failed")):
            with self.assertRaises(OSError):
                mr.refresh()
        projected = [
            p
            for p in self.captures.rglob("*.md")
            if not p.name.startswith("2026-08-25--direct")
        ]
        self.assertEqual(projected, [])
        self.assertEqual(self._temp_residue(), [])


if __name__ == "__main__":
    unittest.main()
