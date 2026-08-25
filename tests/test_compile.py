"""Tests for Capture Record compilation (scripts/compile.py).

Subprocess and filesystem are mocked: records live in tmp_path fixtures,
``subprocess.run`` is replaced by a recorder, and the projection/index helpers
are stubbed out. Run with pytest from the repository root.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import subprocess  # noqa: F401 - imported to document what is mocked
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import capture_records as capture_records_mod
import compile as compile_mod

MODEL_A, MODEL_B = compile_mod.COMPILE_MODELS


# ── Fixtures ──────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class FakeRecord:
    """Shape-compatible stand-in for capture_records.CaptureRecord."""

    path: Path
    record_id: str
    record_day: str
    capture_status: str = "narrative"
    payload_sha256: str = "payload-default"
    source_sid: str | None = None
    body: str = ""


def write_record_file(
    root: Path,
    day: str,
    name: str,
    *,
    status: str = "narrative",
    aggregate_sha: str | None = None,
    source_aggregate_file: str | None = None,
) -> Path:
    directory = root / "records" / "captures" / day
    directory.mkdir(parents=True, exist_ok=True)
    fields = [
        'record_type: "capture_record"',
        f"record_day: {day}",
        f'capture_status: "{status}"',
        f'payload_sha256: "{name}-payload"',
    ]
    if aggregate_sha is not None:
        fields.append(f'source_aggregate_sha256: "{aggregate_sha}"')
    if source_aggregate_file is not None:
        fields.append(f'source_aggregate_file: "{source_aggregate_file}"')
    path = directory / name
    path.write_text("---\n" + "\n".join(fields) + "\n---\n\nbody text\n", encoding="utf-8")
    return path


def make_record(path: Path, day: str, *, status: str = "narrative") -> FakeRecord:
    return FakeRecord(
        path=path,
        record_id=f'capture:["{day}","{path.name}"]',
        record_day=day,
        capture_status=status,
        payload_sha256=f"{path.name}-payload",
        body="SECRET-BODY-TEXT",
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated compile environment: fake records, state, and subprocess."""
    state = {
        "ingested": {},
        "compiled_captures": {},
        "compiled_raw_inputs": {},
    }
    records: list[FakeRecord] = []
    tag_calls: list[dict] = []
    subprocess_calls: list[list[str]] = []
    outcomes: list[tuple[int, str]] = []
    discovered_days: list[str] = []

    def fake_discover(record_day: str, current_state: dict) -> tuple[Path, ...]:
        discovered_days.append(record_day)
        day_dir = tmp_path / "raw" / record_day
        day_dir.mkdir(parents=True, exist_ok=True)
        sample = day_dir / "clip.md"
        sample.write_text(f"raw-{record_day}", encoding="utf-8")
        return (sample,)

    def fake_tag(paths, *, compiled: bool, capture_id: str | None = None) -> None:
        tag_calls.append(
            {"paths": tuple(paths), "compiled": compiled, "capture_id": capture_id}
        )

    def fake_run(command, **kwargs):
        subprocess_calls.append(list(command))
        code, output = outcomes.pop(0) if outcomes else (0, "")
        return SimpleNamespace(returncode=code, stdout=output, stderr="")

    monkeypatch.setattr(compile_mod, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(compile_mod, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(compile_mod, "LOCK_FILE", tmp_path / ".compile.lock")
    monkeypatch.setattr(compile_mod, "list_capture_records", lambda: tuple(records))
    monkeypatch.setattr(
        compile_mod,
        "read_capture",
        lambda target: next(r for r in records if r.path == Path(target)),
    )
    monkeypatch.setattr(compile_mod, "discover_unprocessed_raw", fake_discover)
    monkeypatch.setattr(compile_mod, "tag_raw_inputs", fake_tag)
    monkeypatch.setattr(compile_mod, "capture_state_key", lambda record: record.record_id)
    monkeypatch.setattr(compile_mod, "load_state", lambda: state)
    monkeypatch.setattr(compile_mod, "normalize_index_file", lambda: False)
    monkeypatch.setattr(compile_mod, "refresh_projection", lambda: None)
    monkeypatch.setattr(compile_mod.subprocess, "run", fake_run)

    return SimpleNamespace(
        root=tmp_path,
        state=state,
        records=records,
        tag_calls=tag_calls,
        subprocess_calls=subprocess_calls,
        outcomes=outcomes,
        discovered_days=discovered_days,
        add_record=lambda *a, **kw: records.append(make_record(*a, **kw)) or records[-1],
    )


def add_pending(env_env, count: int, day: str = "2026-08-25"):
    for i in range(count):
        path = write_record_file(env_env.root, day, f"{day}--session--{i:03d}.md")
        env_env.add_record(path, day)


# ── Assignment balance N=1..7 ─────────────────────────────────────────


@pytest.mark.parametrize("count", range(1, 8))
def test_assignments_are_balanced_for_n_records(env, count):
    add_pending(env, count)
    pending = compile_mod.select_pending(tuple(env.records), env.state)
    assignments = compile_mod.assign_models(pending)

    assert len(assignments) == count
    counts = {model: 0 for model in compile_mod.COMPILE_MODELS}
    for index, (_, model) in enumerate(assignments):
        assert model == compile_mod.COMPILE_MODELS[index % len(compile_mod.COMPILE_MODELS)]
        counts[model] += 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_ordering_is_stable_and_sorted(env):
    days = ["2026-08-26", "2026-08-24", "2026-08-25"]
    for day in days:
        add_pending(env, 2, day=day)
    env.records.reverse()  # shuffle input order

    first = [r.record_id for r in compile_mod.select_pending(tuple(env.records), env.state)]
    second = [r.record_id for r in compile_mod.select_pending(tuple(env.records), env.state)]
    assert first == second
    keys = [(_day(r), r.path.name) for r in compile_mod.select_pending(tuple(env.records), env.state)]
    assert keys == sorted(keys)


def _day(record: FakeRecord) -> str:
    return str(record.record_day)[:10]


# ── Selection exclusions ──────────────────────────────────────────────


def test_excludes_no_op_failure_and_already_compiled(env):
    add_pending(env, 3)
    no_op_path = write_record_file(env.root, "2026-08-25", "2026-08-25--no-op.md", status="no_op")
    failure_path = write_record_file(env.root, "2026-08-25", "2026-08-25--failure.md", status="failure")
    env.add_record(no_op_path, "2026-08-25", status="no_op")
    env.add_record(failure_path, "2026-08-25", status="failure")

    done = env.records[0]
    env.state["compiled_captures"][done.record_id] = {
        "payload_sha256": done.payload_sha256,
        "compiled_at": "earlier",
        "model": MODEL_A,
    }

    selected = compile_mod.select_pending(tuple(env.records), env.state)
    ids = {r.record_id for r in selected}
    assert done.record_id not in ids
    assert all(r.capture_status == "narrative" for r in selected)
    assert len(selected) == 2


def test_changed_payload_is_retried(env):
    add_pending(env, 1)
    record = env.records[0]
    env.state["compiled_captures"][record.record_id] = {
        "payload_sha256": "stale-payload",
        "compiled_at": "earlier",
        "model": MODEL_A,
    }
    assert compile_mod.select_pending(tuple(env.records), env.state) == [record]


def test_all_flag_forces_recompile_of_compiled_records(env, capsys):
    add_pending(env, 1)
    record = env.records[0]
    env.state["compiled_captures"][record.record_id] = {
        "payload_sha256": record.payload_sha256,
        "compiled_at": "earlier",
        "model": MODEL_A,
    }
    assert compile_mod.main(["--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert record.path.name in out


# ── Legacy bootstrap ──────────────────────────────────────────────────


def test_legacy_bootstrap_marks_matching_history_compiled(env):
    legacy_hash = "04e2a1dfd7e03626"
    matched_path = write_record_file(
        env.root,
        "2026-08-24",
        "2026-08-24--matched.md",
        aggregate_sha=legacy_hash + "1e374703af600463166cf4fe7d2f7f2f8a1037579b3fedcc",
        source_aggregate_file="daily/2026-08-24.md",
    )
    unmatched_path = write_record_file(
        env.root,
        "2026-08-24",
        "2026-08-24--unmatched.md",
        aggregate_sha="ffffffff" + "tail",
        source_aggregate_file="daily/2026-08-24.md",
    )
    env.add_record(matched_path, "2026-08-24")
    env.add_record(unmatched_path, "2026-08-24")
    env.state["ingested"]["2026-08-24.md"] = {"hash": legacy_hash}

    bootstrapped = compile_mod.bootstrap_legacy_compiles(tuple(env.records), env.state)
    assert bootstrapped == 1
    entry = env.state["compiled_captures"][env.records[0].record_id]
    assert entry["model"] == compile_mod.LEGACY_BOOTSTRAP_MODEL
    assert entry["payload_sha256"] == env.records[0].payload_sha256

    remaining = compile_mod.select_pending(tuple(env.records), env.state)
    assert [r.record_id for r in remaining] == [env.records[1].record_id]


# ── Raw Input attachment ──────────────────────────────────────────────


def test_same_day_raw_inputs_attach_to_first_pending_record_only(env):
    add_pending(env, 3, day="2026-08-25")

    attachments = compile_mod.plan_raw_attachments(tuple(env.records), env.state)
    assert env.discovered_days == ["2026-08-25"]  # discovered exactly once

    attached_keys = [key for key, paths in attachments.items() if paths]
    assert attached_keys == [env.records[0].record_id]
    assert attachments[env.records[1].record_id] == ()
    assert attachments[env.records[2].record_id] == ()


def test_different_days_each_get_their_own_group(env):
    add_pending(env, 1, day="2026-08-24")
    add_pending(env, 1, day="2026-08-25")
    attachments = compile_mod.plan_raw_attachments(tuple(env.records), env.state)
    assert sorted(env.discovered_days) == ["2026-08-24", "2026-08-25"]
    assert all(len(paths) == 1 for paths in attachments.values())


# ── Prompt and command shape ──────────────────────────────────────────


def test_prompt_starts_with_exact_prefix_and_names_one_record(env):
    add_pending(env, 1)
    record = env.records[0]
    raw_paths = (env.root / "raw" / "clip.md",)
    prompt = compile_mod.build_prompt(record, raw_paths)

    assert prompt.startswith("/skill:unslop")
    assert prompt.count(str(record.path)) == 1
    assert str(raw_paths[0]) in prompt
    assert "knowledge/index.md" in prompt  # told to read the index first
    assert record.body not in prompt  # never embeds bodies


def test_omp_command_matches_contract_exactly():
    prompt = "/skill:unslop\nbody"
    vault_root = "/some/vault"

    original_root = compile_mod.ROOT_DIR
    try:
        compile_mod.ROOT_DIR = Path(vault_root)
        command = compile_mod.build_command(MODEL_A, prompt)
    finally:
        compile_mod.ROOT_DIR = original_root

    assert command[:8] == [
        "omp",
        "-p",
        "--no-session",
        "--auto-approve",
        "--cwd",
        vault_root,
        "--model",
        MODEL_A,
    ]
    assert command[8] == prompt
    assert compile_mod.COMPILE_MODELS == (
        "anthropic/claude-sonnet-5",
        "openai-codex/gpt-5.6-terra",
    )


# ── End-to-end success / failure transitions ──────────────────────────


def test_successful_compile_persists_state_and_compiled_tags(env):
    add_pending(env, 2)
    assert compile_mod.main([]) == 0

    compiled = env.state["compiled_captures"]
    assert set(compiled) == {r.record_id for r in env.records}
    for record in env.records:
        assert compiled[record.record_id]["payload_sha256"] == record.payload_sha256
        assert compiled[record.record_id]["model"] in compile_mod.COMPILE_MODELS

    raw_rel = f"raw/{_day(env.records[0])}/clip.md"
    assert raw_rel in env.state["compiled_raw_inputs"]
    raw_entry = env.state["compiled_raw_inputs"][raw_rel]
    assert raw_entry["capture_id"] == env.records[0].record_id
    assert raw_entry["sha256"] == hashlib.sha256(b"raw-2026-08-25").hexdigest()

    compiled_flags = [call["compiled"] for call in env.tag_calls]
    assert compiled_flags.count(True) == 1  # only the first record owns raws
    assert compiled_flags.count(False) == 1

    # Prompt travelled through argv, never through stdin; no article bodies.
    for call in env.subprocess_calls:
        assert call[:1] == ["omp"]
        assert "-p" in call
        assert call[-1].startswith("/skill:unslop")
        assert all("SECRET-BODY-TEXT" not in part for part in call)

    saved = json.loads((env.root / "state.json").read_text(encoding="utf-8"))
    assert set(saved["compiled_captures"]) == {r.record_id for r in env.records}


def test_failed_invocation_marks_nothing_compiled_and_exits_nonzero(env):
    add_pending(env, 2)
    env.outcomes.append((1, "agent exploded"))
    assert compile_mod.main([]) == 1

    failed_record, ok_record = env.records[0], env.records[1]
    assert failed_record.record_id not in env.state["compiled_captures"]
    assert ok_record.record_id in env.state["compiled_captures"]

    # Raw inputs stay uncompiled: tagged before the attempt, never marked compiled.
    assert all(call["compiled"] is False for call in env.tag_calls)
    assert env.state["compiled_raw_inputs"] == {}

    saved = json.loads((env.root / "state.json").read_text(encoding="utf-8"))
    assert failed_record.record_id not in saved.get("compiled_captures", {})


def test_file_flag_compiles_single_record(env, capsys):
    add_pending(env, 2)
    target = env.records[1].path
    assert compile_mod.main(["--file", str(target)]) == 0
    assert set(env.state["compiled_captures"]) == {env.records[1].record_id}
    assert env.discovered_days == []  # --file skips raw attachment planning


def test_dry_run_changes_nothing(env, capsys):
    add_pending(env, 2)
    assert compile_mod.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out
    assert env.subprocess_calls == []
    assert env.state["compiled_captures"] == {}
    assert env.tag_calls == []
    assert not (env.root / "state.json").exists()


# ── Regressions: final-tag hashing and lock-ordered state load ────────


def test_successfully_tagged_raw_is_absent_from_next_discovery(env, monkeypatch):
    """State must hash the final tagged bytes so discovery excludes them."""
    day = "2026-08-25"
    add_pending(env, 1, day=day)
    raw_dir = env.root / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "clip.md"
    raw_file.write_text(
        "---\n"
        'record_type: "raw_input"\n'
        f"created: {day}T10:00:00+00:00\n"
        "---\n\nraw body\n",
        encoding="utf-8",
    )

    # Wire the REAL tagging and discovery through, pointed at the tmp vault.
    monkeypatch.setattr(compile_mod, "tag_raw_inputs", capture_records_mod.tag_raw_inputs)
    monkeypatch.setattr(
        compile_mod, "discover_unprocessed_raw", capture_records_mod.discover_unprocessed_raw
    )
    monkeypatch.setattr(capture_records_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        capture_records_mod, "CAPTURE_LOCK_FILE", env.root / ".capture-records.lock"
    )

    assert compile_mod.main([]) == 0

    saved = json.loads((env.root / "state.json").read_text(encoding="utf-8"))
    raw_entry = saved["compiled_raw_inputs"]["raw/clip.md"]
    # The stored hash covers exactly the bytes now on disk, tags included.
    assert raw_entry["sha256"] == hashlib.sha256(raw_file.read_bytes()).hexdigest()
    assert raw_entry["capture_id"] == env.records[0].record_id

    # Regression: hashing before the final tag write made discovery return
    # this already-compiled input again on the next run.
    assert capture_records_mod.discover_unprocessed_raw(day, saved) == ()


def test_second_serialized_invocation_loads_state_only_after_the_lock(env, monkeypatch):
    """A queued invocation must not read state until it holds the compile lock."""
    add_pending(env, 1)

    # First invocation compiles and persists state.
    assert compile_mod.main([]) == 0

    snapshots: list[set[str]] = []

    def disk_load_state():
        saved = json.loads((env.root / "state.json").read_text(encoding="utf-8"))
        snapshots.append(set(saved.get("compiled_captures", {})))
        return saved

    monkeypatch.setattr(compile_mod, "load_state", disk_load_state)

    holder = open(env.root / ".compile.lock", "w")  # noqa: SIM115 - closed in finally
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

        box: dict[str, int] = {}
        runner = threading.Thread(
            target=lambda: box.update(code=compile_mod.main([])), daemon=True
        )
        runner.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and runner.is_alive() and not snapshots:
            time.sleep(0.01)

        # Still blocked outside the lock: no snapshot was loaded yet.
        assert snapshots == []
        assert runner.is_alive()

        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        runner.join(timeout=10)
    finally:
        holder.close()

    assert box.get("code") == 0
    # The queued invocation loaded exactly the first invocation's snapshot.
    assert snapshots == [{env.records[0].record_id}]
