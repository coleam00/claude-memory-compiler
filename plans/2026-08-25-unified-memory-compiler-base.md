# Unified Memory Compiler Base

## Goal

Replace the unwieldy index, compilation log, and daily-log projections with one Obsidian Base whose views expose raw inputs, capture records, compilation runs, and knowledge articles through their own schemas.

## Decisions

- Use one Base with schema-specific views.
- Include all 359 Markdown files under `raw/`; distinguish the 285 high-confidence Web Clipper notes in a dedicated view.
- Preserve all 3,371 aggregate records: 3,290 capture records and 81 compilation runs.
- Keep each migrated record body byte-for-byte equivalent to its source payload.
- Keep `daily/*.md`, `knowledge/index.md`, `knowledge/log.md`, current state hashes, and runtime readers/writers authoritative during this Base-first phase.

## Note contracts

### Capture Record

Required properties:

- `record_type`: `capture_record`
- `record_id`: string
- `record_day`: date
- `capture_kind`: `session | memory_flush`
- `capture_status`: `narrative | no_op | failure`
- `source_timestamp`: string
- `source_sid`: string or null
- `source_file`: string
- `source_aggregate_file`: string
- `source_ordinal`: integer
- `source_line_start`: integer
- `source_line_end`: integer
- `payload_sha256`: string
- `source_aggregate_sha256`: string
- `tags`: `[memory-compiler/capture-record]`

Body invariant: exact original record heading, payload, whitespace, links, and trailing separator.

Provenance: `source_file` is the logical compiler source daily note; `source_aggregate_file` is that same daily note, whose exact byte slice and aggregate hash back the record.

### Compilation Run

Required properties:

- `record_type`: `compilation_run`
- `record_id`: string
- `operation`: `compile | query_filed`
- `started_at`: datetime
- `source_file`: string
- `source_aggregate_file`: string
- `source_day`: date
- `run_qualifier`: string or null
- `change_kind`: `created_and_updated | created_only | updated_only | no_article_change`
- `articles_created`: list of strings
- `articles_updated`: list of strings
- `source_line_start`: integer
- `source_line_end`: integer
- `payload_sha256`: string
- `source_aggregate_sha256`: string
- `tags`: `[memory-compiler/compilation-run]`

Body invariant: exact original run entry, including extension fields and narrative.

Provenance: `source_file` is the logical compiler source daily note; `source_aggregate_file` is `knowledge/log.md`, whose exact byte slice and aggregate hash back the run.

## Implementation

1. Materialize capture records under `records/captures/<day>/` with deterministic IDs and filenames.
2. Materialize compilation runs under `records/runs/<source-day>/` with deterministic IDs and filenames.
3. Refactor the existing knowledge Base into one `Memory Compiler.base` with schema-specific views for knowledge, raw inputs, Web Clippings, capture narratives, capture failures, capture no-ops, and compilation runs.
4. Add a dashboard note embedding the Base.
5. Validate exact counts, required property types, enum values, source hashes, payload parity, YAML structure, Base filtering, and Obsidian rendering.
6. Keep the projection live: `flush.append_to_daily_log`, `compile.compile_daily_log` (post-normalize), and `query.run_query --file-back` call `materialize_records.refresh_projection()` after their aggregate writes. The refresh is best-effort and never fails the writer; a skipped refresh self-heals on the next run or a manual `uv run python scripts/materialize_records.py`.

## Blast radius

This phase is additive. Aggregate content, whole-file hash state, compilation selection, SessionStart injection, index normalization, and log-writing prompts are unchanged; the only writer change is the appended best-effort projection refresh. Concurrent flushes may race the refresh, which is safe because materialization is deterministic and idempotent. The aggregates remain deletion blockers until the later OMP/compiler cutover migrates every caller.
